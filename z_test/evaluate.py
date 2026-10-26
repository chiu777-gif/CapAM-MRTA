"""Evaluate a checkpoint on generated files or original TAPTC scenarios."""
import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
from eval import evaluate as evaluate_scenarios
from problems.mrta.config import DATA_FORMATS, DATA_VERSION, DEFAULT_CONFIG, DEFAULT_SEED, MRTAConfig
from problems.mrta.data import collate_mrta, load_file_dataset
from utils import load_model, move_to


def resolve_checkpoint(path):
    path = Path(path).resolve()
    if path.is_dir():
        candidates = [(int(match.group(1)), item) for item in path.glob('epoch-*.pt')
                      for match in [re.fullmatch(r'epoch-(\d+)\.pt', item.name)] if match]
        if not candidates:
            raise FileNotFoundError('No epoch checkpoints in {}'.format(path))
        path = max(candidates)[1]
    if not path.is_file():
        raise FileNotFoundError('Checkpoint does not exist: {}'.format(path))
    return path


def expand_sources(paths):
    sources = []
    for value in paths:
        path = Path(value).resolve()
        if path.is_dir() and not list(path.glob('group[12]/r*.txt')):
            # Generated .pkl directories and an individual TAPTC group are accepted too.
            files = sorted(path.glob('*.pkl')) or sorted(path.glob('r*.txt'))
            if not files:
                raise ValueError('No generated .pkl files or TAPTC cases in {}'.format(path))
            sources.extend(files)
        else:
            sources.append(path)
    return sources


def summarize(rows):
    total = sum(row['task_count'] for row in rows)
    done = sum(row['tasks_done'] for row in rows)
    seconds = sum(row['amortized_duration_seconds'] for row in rows)
    return {
        'scenarios': len(rows), 'completed_tasks': done, 'total_tasks': total,
        'completion_rate': done / total, 'completion_percent': 100. * done / total,
        'mean_scenario_completion_rate': sum(row['completion_rate'] for row in rows) / len(rows),
        'mean_cost': sum(row['cost'] for row in rows) / len(rows),
        'inference_seconds_total': seconds,
        'inference_seconds_per_scenario': seconds / len(rows),
    }


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', required=True, help='Checkpoint file, or directory for its latest epoch')
    parser.add_argument('--data', '--datasets', dest='datasets', nargs='+', required=True,
                        help='Generated .pkl files/directory, or original TAPTC root/group/task files')
    parser.add_argument('--output_dir', type=Path, help='Default: z_test/results/<checkpoint>_<UTC timestamp>')
    parser.add_argument('--eval_batch_size', '--batch_size', type=int, default=1,
                        help='Use 1 to measure single-scenario latency; larger values measure batched throughput')
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    device_group.add_argument('--no_cuda', dest='device', action='store_const', const='cpu')
    parser.add_argument('--cpu_threads', type=int, default=4)
    parser.add_argument('--warmup_batches', type=int, default=1,
                        help='Untimed warmup repetitions of the first batch')
    parser.add_argument('--decode_strategy', choices=['greedy', 'sample'], default='greedy')
    parser.add_argument('--width', type=int, default=1, help='Candidate solutions per scenario for sample decoding')
    parser.add_argument('--max_calc_batch_size', type=int, default=4096)
    parser.add_argument('--softmax_temperature', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--data_format', choices=DATA_FORMATS, default='auto')
    parser.add_argument('--agent_max_speed', type=float,
                        help='Physical speed for original TAPTC/untagged files; generated files retain their saved speed')
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--num_samples', '--val_size', type=int, help='Evaluate this many scenarios; default all')
    parser.add_argument('--no_progress_bar', action='store_true')
    parser.add_argument('--overwrite', '-f', action='store_true')
    opts = parser.parse_args(args)
    if min(opts.eval_batch_size, opts.cpu_threads, opts.width) < 1:
        parser.error('eval_batch_size, cpu_threads and width must be positive')
    if opts.warmup_batches < 0 or opts.offset < 0 or (opts.num_samples is not None and opts.num_samples < 1):
        parser.error('warmup_batches/offset must be nonnegative; num_samples must be positive')
    if opts.max_calc_batch_size < opts.eval_batch_size:
        parser.error('max_calc_batch_size must be >= eval_batch_size')
    if not math.isfinite(opts.softmax_temperature) or opts.softmax_temperature <= 0:
        parser.error('softmax_temperature must be finite and positive')
    if not 0 <= opts.seed < 2 ** 63:
        parser.error('seed must be in [0, 2**63)')
    if opts.decode_strategy == 'greedy' and opts.width != 1:
        parser.error('--width applies to sample decoding; leave it at 1 for greedy')

    try:
        checkpoint = resolve_checkpoint(opts.model)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output = opts.output_dir or Path(__file__).resolve().parent / 'results' / (checkpoint.stem + '_' + stamp)
        for name in ('summary.json', 'scenarios.csv', 'batches.csv', 'groups.csv'):
            if (output / name).exists() and not opts.overwrite:
                raise FileExistsError('Already exists: {}. Choose another output_dir or use --overwrite.'.format(output / name))
        torch.set_num_threads(opts.cpu_threads)
        torch.manual_seed(opts.seed)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if opts.device == 'auto' else torch.device(opts.device)
        if device.type not in ('cpu', 'cuda'):
            raise ValueError('Supported devices: cpu and cuda:N')
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable; use --device cpu')
        model, saved = load_model(checkpoint)
        config = MRTAConfig.from_options(SimpleNamespace(**saved)) if saved.get('schema_version') == DATA_VERSION else DEFAULT_CONFIG
        if opts.agent_max_speed is not None:
            config = replace(config, agent_max_speed=opts.agent_max_speed)
        sources = expand_sources(opts.datasets)
        dataset = load_file_dataset(sources, config=config, data_format=opts.data_format,
                                    offset=opts.offset, num_samples=opts.num_samples)
        task_counts = Counter(item['loc'].size(0) for item in dataset)
        if len(task_counts) > 1 and opts.eval_batch_size != 1:
            raise ValueError('Mixed task sizes require --eval_batch_size 1, or evaluate each size separately')
        model.to(device).eval()
        model.set_decode_type('greedy' if opts.decode_strategy == 'greedy' else 'sampling',
                              temp=opts.softmax_temperature)
        if opts.warmup_batches:
            first = move_to(collate_mrta([dataset[i] for i in range(min(len(dataset), opts.eval_batch_size))]), device)
            with torch.no_grad():
                for _ in range(opts.warmup_batches):
                    model.sample_many(first, batch_rep=1)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            del first
        # Warmup must not change which random candidate solutions get evaluated.
        torch.manual_seed(opts.seed)
        opts.model = str(checkpoint)
        started = time.perf_counter()
        results = evaluate_scenarios(model, dataset, opts, opts.width, device)
        wall_seconds = time.perf_counter() - started
        rows, batches = [], []
        grouped = defaultdict(list)
        for index, result in enumerate(results):
            row = {key: result[key] for key in ('instance', 'task_count', 'n_agents', 'tasks_done',
                                               'completion_rate', 'cost', 'batch_duration_seconds',
                                               'amortized_duration_seconds')}
            row['completion_percent'] = 100. * row['completion_rate']
            row['batch'] = index // opts.eval_batch_size
            match = re.fullmatch(r'r([12])([1-4])a\d+i\d+(?:_n_\d+)?', Path(row['instance']).stem)
            row['deadline_group'] = ('tight' if match.group(1) == '1' else 'slack') if match else ''
            row['sampled_deadline_fraction'] = int(match.group(2)) / 4 if match else ''
            rows.append(row)
            grouped[(row['task_count'], row['n_agents'], row['deadline_group'], row['sampled_deadline_fraction'])].append(row)
        for start in range(0, len(rows), opts.eval_batch_size):
            batch_rows = rows[start:start + opts.eval_batch_size]
            batches.append({'batch': batch_rows[0]['batch'], 'scenarios': len(batch_rows),
                            'total_tasks': sum(row['task_count'] for row in batch_rows),
                            'completion_rate': summarize(batch_rows)['completion_rate'],
                            'inference_seconds': batch_rows[0]['batch_duration_seconds']})
        groups = [dict(task_count=key[0], n_agents=key[1], deadline_group=key[2],
                       sampled_deadline_fraction=key[3], **summarize(items))
                  for key, items in sorted(grouped.items())]
        summary = {
            'checkpoint': str(checkpoint), 'data_sources': [str(path) for path in sources],
            'encoder': model.encoder, 'n_encode_layers': model.n_encode_layers,
            'evaluated_at_utc': datetime.now(timezone.utc).isoformat(),
            'device': str(device), 'cpu_threads': torch.get_num_threads(),
            'decode_strategy': opts.decode_strategy, 'sampling_width': opts.width if opts.decode_strategy == 'sample' else 0,
            'seed': opts.seed, 'eval_batch_size': opts.eval_batch_size, 'warmup_batches': opts.warmup_batches,
            'offset': opts.offset, 'num_samples_requested': opts.num_samples,
            'units': {'coordinate_scale': config.coordinate_scale, 'workload_scale': config.workload_scale,
                      'physical_speed_for_untagged_files': config.agent_max_speed},
            'task_count_histogram': dict(sorted(task_counts.items())),
            'robot_count_histogram': dict(sorted(Counter(row['n_agents'] for row in rows).items())),
            **summarize(rows),
            'inference_batches': len(batches),
            'mean_batch_inference_seconds': sum(row['inference_seconds'] for row in batches) / len(batches),
            'evaluation_wall_seconds': wall_seconds,
            'timing_definition': 'Full-solution policy inference including simulation and candidate selection; '
                                 'excludes warmup, loading, input transfer, replay verification, and output writing. '
                                 'Per-scenario time is amortized when eval_batch_size > 1.',
            'by_robot_count': {robots: summarize([row for row in rows if row['n_agents'] == robots])
                               for robots in sorted({row['n_agents'] for row in rows})},
        }
        output.mkdir(parents=True, exist_ok=True)
        write_csv(output / 'scenarios.csv', rows)
        write_csv(output / 'batches.csv', batches)
        write_csv(output / 'groups.csv', groups)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print('Encoder: {}; layers: {}'.format(summary['encoder'], summary['n_encode_layers']))
    print('Scenarios: {}; tasks: {}; completed on time: {}'.format(
        summary['scenarios'], summary['total_tasks'], summary['completed_tasks']))
    print('Task completion rate: {:.2f}%'.format(summary['completion_percent']))
    print('Mean cost: {:.6f}'.format(summary['mean_cost']))
    print('Total inference time: {:.6f} s'.format(summary['inference_seconds_total']))
    print('Mean inference time per scenario{}: {:.3f} ms'.format(
        ' (amortized across batch)' if opts.eval_batch_size > 1 else '',
        1000 * summary['inference_seconds_per_scenario']))
    print('Robot count | Scenarios | Completion')
    for robots, stats in summary['by_robot_count'].items():
        print('{:11d} | {:9d} | {:.2f}%'.format(robots, stats['scenarios'], stats['completion_percent']))
    print('Results: {}'.format(output.resolve()))
    return summary


if __name__ == '__main__':
    main()
