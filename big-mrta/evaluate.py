"""Evaluate BiG-MRTA completion incentives or adapted local cost matching."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent


def stable_process():
    # The upstream Hungarian helper turns robot names into a set. Fix hash
    # randomization before Python starts, so row order/tie-breaking is repeatable.
    if os.environ.get('PYTHONHASHSEED') != '0':
        env = dict(os.environ, PYTHONHASHSEED='0')
        os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve())] + sys.argv[1:], env)


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(handle.read()).hexdigest()


def dataset_fingerprint(path):
    """Hash only evaluated TAPTC tasks and their paired agent files.

    Directory extras such as n_200, demo files and converted indices are not
    evaluation inputs. A versioned pickle retains its original file SHA256.
    """
    path = Path(path).resolve()
    if path.is_file() and path.suffix != '.txt':
        digest = sha256(path)
        return dict(kind='file_sha256', sha256=digest,
                    files=[dict(path=path.name, sha256=digest)])
    tasks = sorted(path.glob('group[12]/r*.txt')) if path.is_dir() else [path]
    if not tasks:
        raise ValueError('No TAPTC task inputs in ' + str(path))
    root = path if path.is_dir() else path.parent.parent
    inputs = set(tasks)
    for task in tasks:
        inputs.add(task.parent.parent / 'agent' / ('a' + task.stem.split('a', 1)[1] + '.txt'))
    records = [dict(path=item.relative_to(root).as_posix(), sha256=sha256(item))
               for item in sorted(inputs)]
    digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(kind='taptc_task_and_agent_manifest_sha256', sha256=digest, files=records)


def upstream_manifest():
    manifest = json.loads((HERE / 'upstream' / 'manifest.json').read_text())
    for name, entry in manifest['files'].items():
        if sha256(HERE / 'upstream' / name) != entry['sha256']:
            raise ValueError('Vendored upstream file has changed: ' + name)
    return manifest


def summarize(rows):
    import numpy as np
    total = sum(row['task_count'] for row in rows)
    done = sum(row['tasks_done'] for row in rows)
    seconds = sum(row['inference_seconds'] for row in rows)
    all_complete = all(row['cost'] is not None for row in rows)
    planning = sum(row.get('path_planning_seconds', 0.) for row in rows)
    return {
        'path_planning_seconds_total': planning,
        'path_planning_seconds_per_scenario': planning / len(rows),
        'path_planning_cpu_seconds_total': sum(row.get('path_planning_cpu_seconds', 0.) for row in rows),
        'cache_read_seconds_total': sum(row.get('cache_read_seconds', 0.) for row in rows),
        'cache_write_seconds_total': sum(row.get('cache_write_seconds', 0.) for row in rows),
        'cache_hits': sum(bool(row.get('cache_hit', False)) for row in rows),
        'cache_misses': sum(bool(row.get('navigation_key')) and not row.get('cache_hit', False) for row in rows),
        'scenarios': len(rows), 'completed_tasks': done, 'total_tasks': total,
        'completion_rate': done / total, 'completion_percent': 100. * done / total,
        'mean_scenario_completion_rate': sum(row['completion_rate'] for row in rows) / len(rows),
        'assigned_tasks': sum(row['assigned_tasks'] for row in rows),
        'unassigned_tasks': sum(row['unassigned_tasks'] for row in rows),
        'late_assigned_tasks': sum(row['late_assigned_tasks'] for row in rows),
        'mean_cost': sum(row['cost'] for row in rows) / len(rows) if all_complete else None,
        'mean_assigned_task_deadline_cost': sum(row['assigned_task_deadline_cost'] for row in rows) / len(rows),
        'inference_seconds_total': seconds, 'inference_seconds_per_scenario': seconds / len(rows),
        'inference_milliseconds_p50': float(np.percentile([row['inference_seconds'] for row in rows], 50) * 1000),
        'inference_milliseconds_p95': float(np.percentile([row['inference_seconds'] for row in rows], 95) * 1000),
        'policy_seconds_total': sum(row['policy_seconds'] for row in rows),
        'policy_seconds_per_scenario': sum(row['policy_seconds'] for row in rows) / len(rows),
        'policy_calls': sum(row['policy_calls'] for row in rows),
        'relaxed_decisions': sum(row['relaxed_decisions'] for row in rows),
        'isolated_relaxed_decisions': sum(row['isolated_relaxed_decisions'] for row in rows),
    }


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(output, summaries):
    lines = ['# BiG-MRTA evaluation', '',
             'Same saved scenarios; robot counts are grouped as stored, not resampled.', '',
             'The exact authors\' TAPTC driver permits unassigned tasks. Its complete-visit '
             'deadline cost is undefined on incomplete solutions (`null`). '
             '`capam_full_visit` is an explicitly modified protocol: StateMRTA continuous '
             'service times and sequential decisions. Completion mode relaxes deadlines only on '
             'empty assignments; cost mode allows late-task edges and minimizes immediate '
             'penalty in each matching. Both visit every task and report the same cost metric as CapAM.', '']
    for protocol, summary in summaries.items():
        lines.extend(['## ' + protocol, '',
                      'Objective: `{}`. {}'.format(summary.get('objective', 'completion'), summary.get('objective_definition', 'Original incentive matching.')), '',
                      'Path worker processes: {} (requested {}).'.format(summary.get('path_workers', 1), summary.get('path_workers_requested', 1)), '',
                      'Planner: `{}`; epsilon: `{}`. Shared preprocessing wall: {:.6f} s.'.format(
                          summary.get('path_planner', 'euclidean'), summary.get('visibility_epsilon'), summary.get('path_preprocessing_wall_seconds', 0.)), '',
                      '| Robots | Scenarios | Mean cost | On-time completion | Full-solution ms/scenario | Planning ms/scenario |',
                      '|---:|---:|---:|---:|---:|---:|'])
        for robots, stats in list(summary['by_robot_count'].items()) + [('All', summary)]:
            cost = '{:.6f}'.format(stats['mean_cost']) if stats['mean_cost'] is not None else 'undefined (unassigned tasks)'
            lines.append('| {} | {} | {} | {:.3f}% | {:.3f} | {:.3f} |'.format(
                robots, stats['scenarios'], cost, stats['completion_percent'],
                stats['inference_seconds_per_scenario'] * 1000, stats.get('path_planning_seconds_per_scenario', 0.) * 1000))
        lines.extend(['', 'Unassigned tasks: {}; relaxed decisions: {}.'.format(
            summary['unassigned_tasks'], summary['relaxed_decisions']), ''])
        lines.extend(['Shared path preprocessing: planning {:.6f} s; cache read {:.6f} s; '
                      'cache write {:.6f} s; hits {}, misses {}.'.format(
                          summary.get('path_planning_seconds_total', 0.),
                          summary.get('cache_read_seconds_total', 0.),
                          summary.get('cache_write_seconds_total', 0.),
                          summary.get('cache_hits', 0), summary.get('cache_misses', 0)), ''])
    lines.extend(['Timing: CPU, batch size 1, serial scenarios; includes matrix/state '
                  'construction, every upstream graph/matching call and simulation. '
                  'Excludes dataset loading/conversion, path preprocessing/cache I/O, warmup, replay checks and output I/O. '
                  'Path planning is shared across protocols, excludes cache I/O, and is zero on cache hits. '
                  'Parallel planning totals sum scenario worker times; preprocessing wall includes worker startup/shutdown. '
                  'Existing CapAM batch-32 amortized times are a different timing regime.', ''])
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main(argv=None):
    from baseline import OBJECTIVES, SOLVERS, prepare, replay_full_visit
    import networkx
    import numpy as np
    import scipy
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits
    from navigation import load_evaluation_data, plan_scenarios
    from z_test_path.path_planner import make_planner
    from z_test_path.visibility_planner import DEFAULT_VISIBILITY_EPSILON

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=HERE.parent / 'z_test/data/n100_train_matched_1000_seed20261024.pkl')
    parser.add_argument('--protocol', choices=['both'] + list(SOLVERS),
                        help='Default: both for completion; capam_full_visit for cost')
    parser.add_argument('--objective', choices=tuple(OBJECTIVES), default='completion',
                        help='Matching objective; cost uses the full-visit adapted protocol')
    parser.add_argument('--output_dir', type=Path, default=HERE / 'results/n100_train_matched_1000_seed20261024')
    parser.add_argument('--cpu_threads', type=int, default=4)
    parser.add_argument('--warmup_scenarios', type=int, default=1)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--path_planner', choices=['auto', 'euclidean', 'visibility_dijkstra'], default='auto')
    parser.add_argument('--visibility_epsilon', type=float, default=DEFAULT_VISIBILITY_EPSILON)
    parser.add_argument('--cache_dir', type=Path, default=HERE.parent / 'z_test_path/cache')
    parser.add_argument('--rebuild_cache', action='store_true')
    parser.add_argument('--path_workers', type=int, default=1,
                        help='Path preprocessing processes; inference remains serial (default: 1)')
    opts = parser.parse_args(argv)
    if opts.path_workers < 1 or opts.cpu_threads < 1 or opts.warmup_scenarios < 0 or opts.offset < 0 or (opts.num_samples is not None and opts.num_samples < 1):
        parser.error('threads/path_workers/num_samples must be positive; warmup/offset nonnegative')
    if opts.protocol is None:
        opts.protocol = 'capam_full_visit' if opts.objective == 'cost' else 'both'
    if opts.objective == 'cost' and opts.protocol != 'capam_full_visit':
        parser.error('cost requires --protocol capam_full_visit so that every task is visited')
    manifest = upstream_manifest()
    data_path = opts.data.resolve()
    dataset, obstacle_map, config = load_evaluation_data(data_path, opts.offset, opts.num_samples)
    if not len(dataset):
        parser.error('No scenarios selected')
    if opts.path_planner == 'auto':
        opts.path_planner = 'visibility_dijkstra' if obstacle_map is not None else 'euclidean'
    if (obstacle_map is not None) != (opts.path_planner == 'visibility_dijkstra'):
        parser.error('Obstacle data requires visibility_dijkstra; visibility_dijkstra requires obstacle data')
    if obstacle_map is not None:
        make_planner(obstacle_map, path_planner=opts.path_planner, visibility_epsilon=opts.visibility_epsilon)
    fingerprint = dataset_fingerprint(data_path)
    protocols = list(SOLVERS) if opts.protocol == 'both' else [opts.protocol]
    output = opts.output_dir.resolve()
    for protocol in protocols:
        for name in ('summary.json', 'scenarios.csv', 'groups.csv', 'routes.jsonl'):
            path = output / protocol / name
            if path.exists() and not opts.overwrite:
                parser.error('Already exists: {}. Use a new output_dir or --overwrite.'.format(path))
    torch.set_num_threads(opts.cpu_threads)
    torch.set_num_interop_threads(1)
    path_workers = min(opts.path_workers, len(dataset)) if obstacle_map is not None else 0
    navigations = [None] * len(dataset)
    planning_rows = [dict(navigation_key=None, cache_hit=False, path_planning_seconds=0.,
                          path_planning_cpu_seconds=0., cache_read_seconds=0., cache_write_seconds=0.)
                     for _ in dataset]
    planning_started = time.perf_counter()
    with threadpool_limits(limits=opts.cpu_threads), torch.no_grad():
        if obstacle_map is not None:
            print('Path preprocessing: {} scenarios, path_workers={}'.format(len(dataset), path_workers), flush=True)
            results = plan_scenarios(dataset, obstacle_map, config, opts.cache_dir,
                                     opts.visibility_epsilon, opts.rebuild_cache, workers=path_workers)
            for completed, (index, tables, stats) in enumerate(results, 1):
                navigations[index], planning_rows[index] = tables, stats
                if completed % 10 == 0 or completed == len(dataset):
                    print('Path preprocessing: {}/{}; cache hits={}'.format(
                        completed, len(dataset), sum(row['cache_hit'] for row in planning_rows)), flush=True)
        scenarios = [prepare(item, coordinate_scale=config.coordinate_scale, workload_scale=config.workload_scale,
                             taptc_source=name if Path(name).suffix == '.txt' else None, navigation=tables)
                     for item, name, tables in zip(dataset, dataset.names, navigations)]
    planning_wall = time.perf_counter() - planning_started if obstacle_map is not None else 0.
    summaries = {}
    cpu_model = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                      if line.startswith('model name')), platform.processor())
    with threadpool_limits(limits=opts.cpu_threads), torch.no_grad():
        for protocol in protocols:
            solve = SOLVERS[protocol]
            for _ in range(opts.warmup_scenarios):
                solve(scenarios[0], objective=opts.objective)
            directory = output / protocol
            directory.mkdir(parents=True, exist_ok=True)
            rows = []
            wall_start = time.perf_counter()
            with (directory / 'routes.jsonl').open('w', encoding='utf-8') as routes:
                for index, scenario in enumerate(scenarios):
                    started = time.perf_counter()
                    result = solve(scenario, objective=opts.objective)
                    duration = time.perf_counter() - started
                    finish = result['finish_times']
                    visited = np.isfinite(finish)
                    completed = visited & (finish <= scenario.deadlines)
                    late = visited & (finish > scenario.deadlines)
                    assigned_cost = float(np.sum(finish[late] / scenario.deadlines[late]))
                    cost = assigned_cost if visited.all() else None
                    tasks_done = int(completed.sum())
                    if protocol == 'capam_full_visit':
                        cost, replay_done = replay_full_visit(scenario, result)
                        if replay_done != tasks_done or not np.isclose(cost, assigned_cost, rtol=2e-6, atol=2e-5):
                            raise AssertionError('Common metrics disagree with independent metrics')
                    row = dict(instance=dataset.names[index], scenario_index=opts.offset + index,
                               task_count=scenario.n_tasks, n_agents=scenario.n_robots,
                               tasks_done=tasks_done, completion_rate=tasks_done / scenario.n_tasks,
                               completion_percent=100. * tasks_done / scenario.n_tasks,
                               assigned_tasks=int(visited.sum()), unassigned_tasks=int((~visited).sum()),
                               late_assigned_tasks=int(late.sum()), cost=cost,
                               assigned_task_deadline_cost=assigned_cost,
                               upstream_travel_cost=result['upstream_travel_cost'],
                               inference_seconds=duration, policy_seconds=result['policy_seconds'],
                               policy_calls=result['policy_calls'], relaxed_decisions=result['relaxed_decisions'],
                               isolated_relaxed_decisions=result['isolated_relaxed_decisions'],
                               objective=opts.objective, path_planner=opts.path_planner, **planning_rows[index])
                    rows.append(row)
                    routes.write(json.dumps(dict(scenario_index=row['scenario_index'], n_agents=scenario.n_robots,
                                                 objective=opts.objective, events=result['events'],
                                                 finish_times=[float(value) if np.isfinite(value) else None for value in finish]),
                                            allow_nan=False) + '\n')
                    if (index + 1) % 25 == 0 or index == len(scenarios) - 1:
                        routes.flush()
                        print('{}: {}/{} scenarios; {:.1f}s wall'.format(
                            protocol, index + 1, len(scenarios), time.perf_counter() - wall_start), flush=True)
            wall_seconds = time.perf_counter() - wall_start
            groups = [dict(n_agents=robots, **summarize([row for row in rows if row['n_agents'] == robots]))
                      for robots in sorted({row['n_agents'] for row in rows})]
            summary = dict(
                algorithm='BiG-MRTA', protocol=protocol, upstream=manifest,
                objective=opts.objective, objective_definition=OBJECTIVES[opts.objective],
                policy_implementation='upstream incentive matching' if opts.objective == 'completion' else 'adapted minimum-added-cost matching',
                path_planner=opts.path_planner,
                visibility_epsilon=opts.visibility_epsilon if obstacle_map is not None else None,
                cache_dir=str(opts.cache_dir.resolve()) if obstacle_map is not None else None,
                path_preprocessing_wall_seconds=planning_wall,
                path_workers=path_workers, path_workers_requested=opts.path_workers,
                path_timing_definition='Precomputed once before warmup and both protocols; shared timings are not additive across protocols. '
                                       'Planning is the sum of scenario worker times (not parallel wall time), excluding cache I/O; cache hits have zero planning time. '
                                       'Preprocessing wall includes worker startup/shutdown, input adaptation and cache I/O.',
                data_sources=[str(data_path)], dataset_sha256=fingerprint['sha256'],
                dataset_fingerprint=fingerprint,
                evaluated_at_utc=datetime.now(timezone.utc).isoformat(),
                device='cpu', cpu_threads=torch.get_num_threads(), eval_batch_size=1,
                parallel_scenarios=1, warmup_scenarios=opts.warmup_scenarios,
                python_hash_seed=os.environ.get('PYTHONHASHSEED'),
                offset=opts.offset, num_samples_requested=opts.num_samples,
                software=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                              networkx=networkx.__version__, torch=torch.__version__),
                hardware=dict(cpu_model=cpu_model, logical_cpus=os.cpu_count(), threadpools=threadpool_info()),
                task_count_histogram=dict(sorted(Counter(row['task_count'] for row in rows).items())),
                robot_count_histogram=dict(sorted(Counter(row['n_agents'] for row in rows).items())),
                parameters=dict(incentive_model=6 if opts.objective == 'completion' else None, matching_algorithm='hungarian', full_task_state=True,
                                initial_range=1e6, initial_payload=100, max_cost=1000,
                                total_time='int(1.1 * max(deadline))', coordinate_scale=config.coordinate_scale, workload_scale=config.workload_scale),
                cost_definition='sum(finish/deadline for finish > deadline), using all N tasks. '
                                'null when any task is unassigned; assigned_task_deadline_cost is not a complete-solution cost.',
                completion_definition='Actual assigned finish <= deadline; includes equality. '
                                      'The upstream driver instead reports its allocation count as numTaskDone.',
                timing_definition='Single-scenario full solution including matrix/state construction, '
                                  'all graph/matching calls and simulation. Excludes loading/conversion, '
                                  'path planning/cache I/O, warmup, replay verification and output I/O. policy_seconds is a separate subset.',
                protocol_changes=[] if protocol == 'upstream_taptc' else [
                    'Use StateMRTA sequential continuous-time dynamics and exact workload/capacity (no ceil).',
                    OBJECTIVES[opts.objective],
                    'Visit every task once; report common replay cost and true deadline completion.',
                ],
                **summarize(rows), evaluation_wall_seconds=wall_seconds,
                by_robot_count={str(group['n_agents']): {key: value for key, value in group.items() if key != 'n_agents'}
                                for group in groups},
            )
            write_csv(directory / 'scenarios.csv', rows)
            write_csv(directory / 'groups.csv', groups)
            (directory / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
            summaries[protocol] = summary
            print('{} ({}): completion={:.3f}%, cost={}, time={:.3f} ms/scenario, unassigned={}'.format(
                protocol, opts.objective, summary['completion_percent'], summary['mean_cost'],
                summary['inference_seconds_per_scenario'] * 1000, summary['unassigned_tasks']), flush=True)
            print('  path planning={:.6f}s total, {:.3f} ms/scenario; cache hits={}, misses={}; cache read={:.6f}s, write={:.6f}s'.format(
                summary['path_planning_seconds_total'], summary['path_planning_seconds_per_scenario'] * 1000,
                summary['cache_hits'], summary['cache_misses'], summary['cache_read_seconds_total'], summary['cache_write_seconds_total']), flush=True)
    # When running protocols separately, include an already-completed companion
    # in the report only when its dataset and scenario selection match.
    for protocol in SOLVERS:
        path = output / protocol / 'summary.json'
        if protocol not in summaries and path.exists():
            companion = json.loads(path.read_text())
            reference = next(iter(summaries.values()))
            if all(companion.get(key) == reference.get(key) for key in
                   ('dataset_sha256', 'offset', 'num_samples_requested', 'cpu_threads', 'warmup_scenarios', 'path_planner', 'visibility_epsilon', 'path_workers')) and companion.get('objective', 'completion') == opts.objective:
                summaries[protocol] = companion
    write_report(output, summaries)
    print('Results: ' + str(output), flush=True)
    return summaries


if __name__ == '__main__':
    stable_process()
    main()
