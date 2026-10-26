"""Evaluate fixed versioned/TAPTC scenarios without constructing training data."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from problems.mrta.config import DATA_VERSION, DEFAULT_SEED, MRTAConfig, add_environment_arguments, add_loading_arguments
from problems.mrta.data import collate_mrta, load_file_dataset
from utils import load_model, move_to
from utils.metrics_logger import CostAccumulator, MetricsLogger, add_metrics_arguments


def evaluate(model, dataset, opts, width, device, metrics_logger=None):
    model.to(device)
    model.eval()
    model.set_decode_type('greedy' if opts.decode_strategy == 'greedy' else 'sampling',
                          temp=opts.softmax_temperature)
    loader = DataLoader(dataset, batch_size=opts.eval_batch_size, collate_fn=collate_mrta)
    results, index = [], 0
    for batch_id, batch in enumerate(tqdm(loader, disable=opts.no_progress_bar)):
        batch = move_to(batch, device)
        batch_size = batch['loc'].size(0)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        best_sequences = best_costs = best_done = None
        remaining = 1 if opts.decode_strategy == 'greedy' else width
        limit = max(1, opts.max_calc_batch_size // batch_size)
        with torch.no_grad():
            while remaining:
                samples = min(remaining, limit)
                sequences, costs, done = model.sample_many(batch, batch_rep=samples)
                done = torch.as_tensor(done, device=device).reshape(-1)
                if best_costs is None:
                    best_sequences, best_costs, best_done = sequences, costs, done
                else:
                    improved = costs < best_costs
                    best_sequences[improved] = sequences[improved]
                    best_done[improved] = done[improved]
                    best_costs = torch.minimum(best_costs, costs)
                remaining -= samples
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        duration = time.perf_counter() - started
        # Replay via the common simulator, independently verify reported metrics,
        # and expose the robot associated with each decision event.
        state = model.problem.replay(batch, best_sequences)
        if not torch.allclose(best_costs, state.get_final_cost()[:, 0]):
            raise RuntimeError('Decoded cost differs from common simulator replay')
        if not torch.equal(best_done, state.tasks_done_success[:, 0]):
            raise RuntimeError('Decoded completion count differs from replay')
        trace = model.problem.make_state(batch)
        robot_sequences = []
        for step in range(best_sequences.size(1)):
            robot_sequences.append(trace.robot_taking_decision[:, 0])
            trace = trace.update(best_sequences[:, step])
        robot_sequences = torch.stack(robot_sequences, dim=1)
        if metrics_logger is not None and metrics_logger.should_log_batch(batch_id + 1):
            stats = CostAccumulator()
            stats.update(best_costs)
            metrics_logger.write('test_batch', batch=batch_id, **stats.summary(),
                                 duration_seconds=duration,
                                 completion_rate=(best_done.float() / batch['loc'].size(1)).mean().item(),
                                 decode_strategy=opts.decode_strategy, width=width, model_path=opts.model)
        for row in range(batch_size):
            task_count = batch['loc'].size(1)
            results.append({
                'instance': dataset.names[index], 'cost': best_costs[row].item(),
                'tasks_done': best_done[row].item(), 'task_count': task_count,
                'completion_rate': best_done[row].item() / task_count,
                'n_agents': batch['n_agents'][row].item(),
                'sequence': best_sequences[row].cpu().tolist(),
                'robot_sequence': robot_sequences[row].cpu().tolist(),
                'batch_duration_seconds': duration,
                'amortized_duration_seconds': duration / batch_size,
            })
            index += 1
    return results


def save_results(results, path, overwrite):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError('Use -f to overwrite {}'.format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == '.csv':
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0]))
            writer.writeheader()
            for result in results:
                row = dict(result)
                for name in ('sequence', 'robot_sequence'):
                    row[name] = json.dumps(row[name])
                writer.writerow(row)
    elif path.suffix == '.json':
        path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    else:
        raise ValueError('Result filename must end in .json or .csv')
    print('Saved {}'.format(path))


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', required=True,
                        help='Versioned .pkl files, TAPTC task .txt files or its group1/group2 directory')
    parser.add_argument('--model', required=True, help='Checkpoint file or model directory containing args.json')
    add_environment_arguments(parser, loading_only=True)
    add_loading_arguments(parser)
    parser.add_argument('--val_size', type=int, default=None, help='Maximum scenarios, default all')
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--decode_strategy', choices=['greedy', 'sample'], default='greedy')
    parser.add_argument('--width', type=int, nargs='+', default=[0], help='0 for greedy; positive sample counts')
    parser.add_argument('--softmax_temperature', type=float, default=1.)
    parser.add_argument('--max_calc_batch_size', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--no_progress_bar', action='store_true')
    parser.add_argument('--results_dir', default='results')
    add_metrics_arguments(parser)
    parser.add_argument('-o', help='Explicit output .json/.csv path, requires one width')
    parser.add_argument('-f', action='store_true', help='Overwrite results')
    opts = parser.parse_args(args)
    if opts.metrics_step < 0:
        parser.error('metrics_step must be nonnegative')
    if opts.o and len(opts.width) != 1:
        parser.error('-o requires one width')
    if opts.eval_batch_size < 1 or opts.max_calc_batch_size < opts.eval_batch_size:
        parser.error('Require 1 <= eval_batch_size <= max_calc_batch_size')
    if not np.isfinite(opts.softmax_temperature) or opts.softmax_temperature <= 0:
        parser.error('softmax_temperature must be finite and positive')
    if (opts.decode_strategy == 'greedy' and opts.width != [0]) or (
            opts.decode_strategy == 'sample' and any(width < 1 for width in opts.width)):
        parser.error('Use --width 0 for greedy, or positive --width values for sample')
    torch.manual_seed(opts.seed)
    model, model_args = load_model(opts.model)
    metrics_run = Path(opts.model).stem + '_' + datetime.now().strftime('%Y%m%dT%H%M%S%f')
    # Raw file unit conversion follows the model's saved physical scales, unless
    # the user explicitly supplies an override. Versioned files carry their own speed.
    import sys
    explicit = list(sys.argv[1:] if args is None else args)
    for name in ('coordinate_scale', 'workload_scale', 'agent_max_speed'):
        if not any(token.split('=')[0] == '--' + name for token in explicit):
            if model_args.get('schema_version') == DATA_VERSION and name in model_args:
                setattr(opts, name, model_args[name])
    try:
        dataset = load_file_dataset(opts.datasets, config=MRTAConfig.from_options(opts),
                                    data_format=opts.data_format, offset=opts.offset, num_samples=opts.val_size)
        device = torch.device('cuda:0' if torch.cuda.is_available() and not opts.no_cuda else 'cpu')
        for width in opts.width:
            path = opts.o or str(Path(opts.results_dir) / '{}_{}_w{}.json'.format(
                Path(opts.model).stem, opts.decode_strategy, width))
            if Path(path).suffix not in ('.csv', '.json'):
                raise ValueError('Result filename must end in .json or .csv')
            if Path(path).exists() and not opts.f:
                raise FileExistsError('Use -f to overwrite {}'.format(path))
            metrics_dir = opts.metrics_dir or str(Path(path).with_suffix('')) + '_metrics'
            print('CSV metrics: {}'.format(metrics_dir))
            with MetricsLogger(metrics_dir, metrics_run, opts.metrics_step) as metrics_logger:
                results = evaluate(model, dataset, opts, width, device, metrics_logger=metrics_logger)
                stats = CostAccumulator()
                stats.update(torch.tensor([row['cost'] for row in results], dtype=torch.float64))
                metrics_logger.write('test_summary', **stats.summary(),
                                     completion_rate=np.mean([row['completion_rate'] for row in results]),
                                     duration_seconds=sum(row['amortized_duration_seconds'] for row in results),
                                     decode_strategy=opts.decode_strategy, width=width, model_path=opts.model)
            save_results(results, path, opts.f)
            print('Scenarios: {}; mean cost: {:.6f}; mean completion: {:.2%}'.format(
                len(results), np.mean([row['cost'] for row in results]),
                np.mean([row['completion_rate'] for row in results])))
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    return results


if __name__ == '__main__':
    main()
