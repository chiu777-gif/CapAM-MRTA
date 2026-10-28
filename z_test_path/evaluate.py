"""Evaluate GCAPCN, AM, or Mamba with cached grid A* or visibility-graph Dijkstra paths."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import torch

from problems.mrta.config import DEFAULT_SEED, MRTAConfig
from problems.mrta.data import collate_mrta, validate_instance
from utils import load_model, move_to
from z_test.evaluate import resolve_checkpoint, write_csv
from z_test_path.inference import evaluate_scenarios, synchronize
from z_test_path.path_adapter import adapt_model
from z_test_path.path_planner import (
    PLANNER_DESCRIPTIONS, get_path, load_obstacle_data, make_planner, planner_settings,
    precompute_navigation, prepare_path_dataset,
)
from z_test_path.reporting import device_metadata, summarize_rows, timing_definitions, write_report
from z_test_path.visibility_planner import DEFAULT_VISIBILITY_EPSILON


def _plan_case(index, item, obstacle_map, extent, subdivisions, cache_dir, rebuild,
               path_planner='grid_astar', visibility_epsilon=DEFAULT_VISIBILITY_EPSILON):
    try:
        item = validate_instance(item)
        navigation, stats = precompute_navigation(item, obstacle_map, extent, subdivisions, cache_dir, rebuild,
                                                  path_planner, visibility_epsilon)
        return index, navigation, stats
    except (ValueError, OSError, KeyError, pickle.UnpicklingError, EOFError) as error:
        raise ValueError('Scenario index {}: {}'.format(index, error)) from error


def _initialize_path_worker():
    torch.set_num_threads(1)


def prepare_navigation(indices, instances, obstacle_map, extent, opts):
    arguments = [(index, item, obstacle_map, extent, opts.path_subdivisions,
                  opts.cache_dir, opts.rebuild_cache, opts.path_planner, opts.visibility_epsilon)
                 for index, item in zip(indices, instances)]
    planned = {}

    def record(result):
        index, navigation, stats = result
        planned[index] = (navigation, stats)
        if not opts.no_progress_bar:
            print('Paths {}/{}: index {}, {}, {:.3f}s planning'.format(
                len(planned), len(indices), index, 'cache hit' if stats['cache_hit'] else 'computed',
                stats['planning_seconds']), flush=True)

    if opts.path_workers == 1:
        for arguments_for_case in arguments:
            record(_plan_case(*arguments_for_case))
    else:
        with ProcessPoolExecutor(max_workers=opts.path_workers,
                                 mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_initialize_path_worker) as pool:
            # Plain arrays avoid PyTorch shared-memory/file-descriptor IPC.
            futures = [pool.submit(_plan_case, values[0],
                                   {key: value.cpu().numpy() for key, value in values[1].items()},
                                   *values[2:]) for values in arguments]
            for future in as_completed(futures):
                record(future.result())
    return ([planned[index][0] for index in indices], [planned[index][1] for index in indices])


def reconstruct_routes(item, result, navigation, obstacle_map):
    """Verify physical polylines and recompute metrics independently in float64."""
    tasks, agents = len(item['loc']), int(item['n_agents'])
    last = [tasks + robot for robot in range(agents)]
    available = np.zeros(agents, dtype=float)
    finish_times = np.zeros(tasks, dtype=float)
    deadlines = item['deadline'].numpy().astype(float)
    workload = item['workload'].numpy().astype(float)
    capacity = item['robots_work_capacity'][:agents].numpy().astype(float)
    speed, extent = float(item['max_speed']), navigation['extent']
    planner = make_planner(obstacle_map, navigation['subdivisions'], navigation['path_planner'],
                           navigation['visibility_epsilon'])
    legs, total_distance = [], 0.
    for task_id, robot in zip(result['sequence'], result['robot_sequence']):
        target = task_id - 1
        if not 0 <= robot < agents:
            raise RuntimeError('Decoded solution selected an inactive robot')
        path = get_path(navigation, last[robot], target)
        for start, goal in zip(path[:-1], path[1:]):
            if not planner.segment_is_free(start / extent, goal / extent):
                raise RuntimeError('Cached route intersects an obstacle')
        np.testing.assert_allclose(path[0], navigation['positions'][last[robot]], atol=1e-10, rtol=0)
        np.testing.assert_allclose(path[-1], navigation['positions'][target], atol=1e-10, rtol=0)
        distance = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        cached = (navigation['robot_to_task'][robot, target] if last[robot] >= tasks
                  else navigation['task_to_task'][last[robot], target])
        if not math.isclose(distance, float(cached), rel_tol=1e-6, abs_tol=1e-7):
            raise RuntimeError('Cached distance differs from route arc length')
        departure = float(available[robot])
        arrival = departure + distance / speed
        finish = arrival + workload[target] / capacity[robot]
        legs.append({'robot_id': robot + 1, 'task_id': task_id,
                     'from_task_id': last[robot] + 1 if last[robot] < tasks else None,
                     'path': path.tolist(), 'distance': distance,
                     'departure_time': departure, 'arrival_time': arrival, 'finish_time': finish,
                     'deadline': float(deadlines[target]), 'on_time': bool(finish <= deadlines[target])})
        available[robot] = finish_times[target] = finish
        last[robot] = target
        total_distance += distance
    done = int((finish_times <= deadlines).sum())
    cost = float(np.where(finish_times > deadlines, finish_times / deadlines, 0.).sum())
    # The policy simulator uses float32. Route arc lengths and this independent
    # replay use float64; tolerate accumulated rounding, not different semantics.
    if done != result['tasks_done'] or not math.isclose(cost, result['cost'], rel_tol=2e-5, abs_tol=1e-3):
        raise RuntimeError('Independent path replay disagrees with decoded cost/completion count')
    return legs, total_distance, abs(cost - result['cost'])


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=Path, required=True, help='Checkpoint file or directory (latest epoch)')
    parser.add_argument('--data', '--dataset', type=Path, required=True, help='Generated obstacle .pkl dataset')
    parser.add_argument('--output_dir', type=Path)
    parser.add_argument('--cache_dir', type=Path, default=Path(__file__).resolve().parent / 'cache')
    parser.add_argument('--path_planner', choices=tuple(PLANNER_DESCRIPTIONS), default='grid_astar',
                        help='Navigation algorithm used for path preprocessing and replay')
    parser.add_argument('--path_subdivisions', type=int, default=10,
                        help='grid_astar only: fine cells along each ASCII cell edge')
    parser.add_argument('--visibility_epsilon', type=float, default=DEFAULT_VISIBILITY_EPSILON,
                        help='visibility_dijkstra only: outward corner offset in the unit [0,1]^2 map')
    parser.add_argument('--path_workers', type=int, default=1, help='Parallel CPU path preprocessing workers')
    parser.add_argument('--rebuild_cache', action='store_true')
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--num_samples', '--val_size', type=int)
    parser.add_argument('--eval_batch_size', '--batch_size', type=int, default=1)
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    device_group.add_argument('--no_cuda', dest='device', action='store_const', const='cpu')
    parser.add_argument('--cpu_threads', type=int, default=1)
    parser.add_argument('--decode_strategy', choices=['greedy', 'sample'], default='greedy')
    parser.add_argument('--width', type=int, default=1)
    parser.add_argument('--max_calc_batch_size', type=int, default=4096)
    parser.add_argument('--softmax_temperature', type=float, default=1.)
    parser.add_argument('--warmup_batches', type=int, default=1)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--no_progress_bar', action='store_true')
    parser.add_argument('--overwrite', '-f', action='store_true')
    opts = parser.parse_args(args)
    if min(opts.path_subdivisions, opts.path_workers, opts.cpu_threads, opts.eval_batch_size,
           opts.width, opts.max_calc_batch_size) < 1:
        parser.error('subdivisions, workers, threads, batch sizes and width must be positive')
    if opts.offset < 0 or opts.warmup_batches < 0 or (opts.num_samples is not None and opts.num_samples < 1):
        parser.error('offset/warmup must be nonnegative; num_samples must be positive')
    if not 0 <= opts.seed < 2 ** 63:
        parser.error('seed must be in [0, 2**63)')
    if opts.decode_strategy == 'greedy' and opts.width != 1:
        parser.error('--width applies to sample decoding')
    if opts.max_calc_batch_size < opts.eval_batch_size:
        parser.error('max_calc_batch_size must be >= eval_batch_size')
    if not math.isfinite(opts.softmax_temperature) or opts.softmax_temperature <= 0:
        parser.error('softmax_temperature must be finite and positive')
    if opts.path_planner == 'visibility_dijkstra' and (
            not math.isfinite(opts.visibility_epsilon) or opts.visibility_epsilon < 4 * np.finfo(float).eps):
        parser.error('visibility_epsilon must be finite and >= 4 * float64 machine epsilon')

    try:
        checkpoint = resolve_checkpoint(opts.model)
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        output = opts.output_dir or Path(__file__).resolve().parent / 'results' / (checkpoint.stem + '_' + stamp)
        for name in ('summary.json', 'overall.csv', 'scenarios.csv', 'groups.csv', 'batches.csv', 'routes.json', 'REPORT.md'):
            if (output / name).exists() and not opts.overwrite:
                raise FileExistsError('Already exists: {}. Use --overwrite.'.format(output / name))
        torch.set_num_threads(opts.cpu_threads)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if opts.device == 'auto' else torch.device(opts.device)
        if device.type not in ('cpu', 'cuda') or (device.type == 'cuda' and not torch.cuda.is_available()):
            raise ValueError('Requested device is unavailable; use --device cpu or an available CUDA device')
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        metadata = device_metadata(device, opts.device, opts.cpu_threads)
        metadata.update(checkpoint=str(checkpoint), dataset=str(opts.data.resolve()),
                        eval_batch_size=opts.eval_batch_size, path_workers=opts.path_workers)
        metadata.update(planner_settings(opts.path_planner, opts.path_subdivisions, opts.visibility_epsilon))
        model, saved = load_model(checkpoint)
        model = adapt_model(model, normalization=saved['normalization'])
        metadata['encoder'] = model.encoder
        model_config = MRTAConfig.from_options(SimpleNamespace(**saved))
        config, obstacle_map, all_instances = load_obstacle_data(opts.data, model_config)
        end = len(all_instances) if opts.num_samples is None else min(len(all_instances), opts.offset + opts.num_samples)
        indices = list(range(opts.offset, end))
        if not indices:
            raise ValueError('Requested dataset slice is empty')
        instances = [all_instances[index] for index in indices]
        if len({len(item['loc']) for item in instances}) > 1 and opts.eval_batch_size != 1:
            raise ValueError('Mixed task sizes require --eval_batch_size 1')
        print('Preparing {} scenarios with {}: workers={}'.format(
            len(instances), opts.path_planner, opts.path_workers), flush=True)
        navigations, path_stats = prepare_navigation(
            indices, instances, obstacle_map, config.map_size / config.coordinate_scale, opts)
        dataset = prepare_path_dataset(instances, indices, navigations, opts.data.resolve())
        model.to(device).eval()
        model.set_decode_type('greedy' if opts.decode_strategy == 'greedy' else 'sampling',
                              temp=opts.softmax_temperature)
        synchronize(device)
        if opts.warmup_batches:
            first = move_to(collate_mrta([dataset[i] for i in range(min(len(dataset), opts.eval_batch_size))]), device)
            with torch.no_grad():
                for _ in range(opts.warmup_batches):
                    model.sample_many(first, batch_rep=1)
            synchronize(device)
        torch.manual_seed(opts.seed)
        opts.model = str(checkpoint)
        results, batches = evaluate_scenarios(model, dataset, opts, opts.width, device)
        route_scenarios, groups = [], defaultdict(list)
        max_replay_error = 0.
        for index, item, navigation, result, stats in zip(indices, instances, navigations, results, path_stats):
            legs, distance, error = reconstruct_routes(item, result, navigation, obstacle_map)
            max_replay_error = max(max_replay_error, error)
            result['index'], result['total_distance'] = index, distance
            result.update(metadata)
            result.update(cache_hit=stats['cache_hit'],
                          path_planning_seconds=stats['planning_seconds'],
                          completion_percent=100. * result['completion_rate'])
            groups[result['n_agents']].append(result)
            route_scenarios.append({'index': index, 'instance': result['instance'],
                                    'task_count': result['task_count'], 'n_agents': result['n_agents'],
                                    'cost': result['cost'], 'completion_rate': result['completion_rate'], 'legs': legs})
        summary = summarize_rows(results, metadata)
        group_rows = [dict(n_agents=count, **summarize_rows(rows, metadata)) for count, rows in sorted(groups.items())]
        for batch in batches:
            batch.update(metadata)
        summary.update({
            'checkpoint': str(checkpoint), 'dataset': str(opts.data.resolve()),
            'device': str(device), 'encoder': model.encoder, 'seed': opts.seed,
            'decode_strategy': opts.decode_strategy, 'width': opts.width,
            'offset': opts.offset, 'eval_batch_size': opts.eval_batch_size,
            'warmup_batches': opts.warmup_batches,
            'path_workers': opts.path_workers,
            'max_independent_replay_cost_error': max_replay_error,
            'total_distance': sum(row['total_distance'] for row in results),
            'distance_units': 'normalized model units',
            'coordinate_scale': config.coordinate_scale,
            'timing_definition': 'Inference includes encoding, decoding and path-table simulation; '
                                 'excludes path preprocessing, warmup, verification and exports',
            'graph_distance': ('sqrt(path_distance**2 + normalized_deadline_difference**2)'
                               if model.encoder == 'gcapcn' else None),
            'encoder_uses_path_distances': model.encoder == 'gcapcn',
            'movement_distance': 'cached collision-free path arc length',
        })
        scalar_rows = [{key: value for key, value in result.items() if key not in ('sequence', 'robot_sequence')}
                       for result in results]
        output.mkdir(parents=True, exist_ok=True)
        write_csv(output / 'scenarios.csv', scalar_rows)
        write_csv(output / 'groups.csv', group_rows)
        write_csv(output / 'batches.csv', batches)
        routes = {
            'format': 'mrta_obstacle_routes', 'version': 1, 'dataset': str(opts.data.resolve()),
            'dataset_sha256': hashlib.sha256(opts.data.read_bytes()).hexdigest(),
            'checkpoint': str(checkpoint), 'coordinate_units': 'normalized',
            'device': str(device), 'encoder': model.encoder,
            **planner_settings(opts.path_planner, opts.path_subdivisions, opts.visibility_epsilon),
            'coordinate_scale': config.coordinate_scale, 'scenarios': route_scenarios,
        }
        (output / 'routes.json').write_text(json.dumps(routes, indent=2) + '\n', encoding='utf-8')
        summary.update(timing_definitions=timing_definitions(),
                       by_n_agents={str(row['n_agents']): row for row in group_rows})
        write_csv(output / 'overall.csv', [{key: value for key, value in summary.items()
                                           if not isinstance(value, (dict, list))}])
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
        write_report(output / 'REPORT.md', summary, group_rows)
    except (ValueError, OSError, KeyError, TypeError, pickle.UnpicklingError, EOFError) as error:
        parser.error(str(error))
    print('Task completion rate: {:.3f}%'.format(summary['completion_percent']))
    print('Mean cost: {:.6f}'.format(summary['mean_cost']))
    print('Device: {} / {} ({})'.format(summary['device_type'], summary['device'], summary['device_name']))
    print('Path planner: {}'.format(summary['planner']))
    print('Path planning: {:.3f}s; cache hits: {}/{}'.format(
        summary['path_planning_seconds_total'], summary['cache_hits'], len(results)))
    print('Policy inference: {:.6f}s total, {:.6f}s/scenario; results: {}'.format(
        summary['inference_seconds_total'], summary['inference_seconds_per_scenario'], output))
    return summary


if __name__ == '__main__':
    main()
