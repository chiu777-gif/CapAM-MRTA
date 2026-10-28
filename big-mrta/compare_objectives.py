"""Run paired completion/cost evaluations with the same full-visit simulator."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def compare(output):
    summaries, rows = {}, {}
    for objective in ('completion', 'cost'):
        directory = output / objective / 'capam_full_visit'
        summaries[objective] = json.loads((directory / 'summary.json').read_text())
        with (directory / 'scenarios.csv').open() as handle:
            rows[objective] = list(csv.DictReader(handle))
    a, b = summaries['completion'], summaries['cost']
    for key in ('dataset_sha256', 'offset', 'num_samples_requested', 'cpu_threads', 'warmup_scenarios',
                'path_planner', 'visibility_epsilon', 'protocol', 'scenarios', 'robot_count_histogram'):
        if a[key] != b[key]:
            raise ValueError('Incompatible objective runs: ' + key)
    if a.get('path_workers', 1) != b.get('path_workers', 1):
        raise ValueError('Incompatible objective runs: path_workers')
    if len(rows['completion']) != len(rows['cost']) or len(rows['completion']) != a['scenarios']:
        raise ValueError('Different scenario row counts')
    paired = []
    for left, right in zip(rows['completion'], rows['cost']):
        for key in ('instance', 'scenario_index', 'n_agents', 'task_count', 'navigation_key'):
            if left[key] != right[key]:
                raise ValueError('Different scenarios/navigation: ' + key)
        paired.append(dict(
            scenario_index=int(left['scenario_index']), n_agents=int(left['n_agents']),
            completion_cost=float(left['cost']), cost_cost=float(right['cost']),
            cost_delta=float(right['cost']) - float(left['cost']),
            completion_rate=float(left['completion_rate']), cost_completion_rate=float(right['completion_rate']),
            completion_delta_pp=float(right['completion_percent']) - float(left['completion_percent']),
            completion_inference_seconds=float(left['inference_seconds']),
            cost_inference_seconds=float(right['inference_seconds'])))
    result = dict(
        data_sources=a['data_sources'], dataset_sha256=a['dataset_sha256'], scenarios=a['scenarios'],
        protocol=a['protocol'], path_planner=a['path_planner'], visibility_epsilon=a['visibility_epsilon'],
        cpu_threads=a['cpu_threads'], warmup_scenarios=a['warmup_scenarios'], path_workers=a.get('path_workers', 1),
        objectives={objective: {key: summary[key] for key in (
            'objective_definition', 'mean_cost', 'completion_percent', 'assigned_tasks', 'unassigned_tasks',
            'inference_seconds_per_scenario', 'path_planning_seconds_total', 'cache_hits', 'cache_misses',
            'cache_read_seconds_total', 'cache_write_seconds_total')} for objective, summary in summaries.items()},
        cost_minus_completion=dict(
            mean_cost=b['mean_cost'] - a['mean_cost'],
            relative_cost_change_percent=100 * (b['mean_cost'] / a['mean_cost'] - 1) if a['mean_cost'] else None,
            completion_percentage_points=b['completion_percent'] - a['completion_percent'],
            scenarios_cost_lower=sum(row['cost_delta'] < -1e-6 for row in paired),
            scenarios_cost_equal=sum(abs(row['cost_delta']) <= 1e-6 for row in paired),
            scenarios_cost_higher=sum(row['cost_delta'] > 1e-6 for row in paired),
        ))
    (output / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    with (output / 'paired_scenarios.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    lines = ['# BiG-MRTA 目标对比', '',
             '数据：`{}`，{} 场景；`capam_full_visit`，CPU {} 线程，预热 {} 次。'.format(
                 a['data_sources'][0], a['scenarios'], a['cpu_threads'], a['warmup_scenarios']), '',
             '路径：`{}`，epsilon={}，规划进程数={}；两个目标逐场景使用相同距离缓存，均访问所有任务。'.format(
                 a['path_planner'], a['visibility_epsilon'], a.get('path_workers', 1)), '',
             '| 配置 | 平均 cost ↓ | 按时完成率 ↑ | 推理 ms/场景 | 规划总秒数 | 缓存命中 |',
             '|---|---:|---:|---:|---:|---:|']
    for objective, summary in summaries.items():
        lines.append('| {} | {:.6f} | {:.3f}% | {:.3f} | {:.6f} | {}/{} |'.format(
            objective, summary['mean_cost'], summary['completion_percent'],
            summary['inference_seconds_per_scenario'] * 1000, summary['path_planning_seconds_total'],
            summary['cache_hits'], summary['scenarios']))
    delta = result['cost_minus_completion']
    lines.extend(['', '`cost` 相对 `completion`：平均 cost 差值 {:+.6f}；按时完成率差值 {:+.3f} 个百分点。'.format(
        delta['mean_cost'], delta['completion_percentage_points']), '',
        '逐场景 cost：{} 个降低，{} 个持平，{} 个升高（绝对容差 1e-6）。'.format(
            delta['scenarios_cost_lower'], delta['scenarios_cost_equal'], delta['scenarios_cost_higher']), '',
        '`completion` 保留原版 deadline 可行边和激励权重，是完成率启发式。`cost` 对所有剩余任务建边，'
        '用 Hungarian 最小化本轮预计新增惩罚：按时为 0，逾期为 finish/deadline；'
        '平局按 SciPy 固定行/任务顺序处理。它是局部匹配，未计入本轮决策对后续任务的影响，'
        '不保证最终总 cost 更低，也不保证完成率更高。两种配置均使用相同 cost 和实际 deadline 统计。', '',
        '推理时间排除规划、缓存 I/O、预热、额外重放及结果写入。两个目标依次运行，避免并发测时干扰；'
        '运行顺序为 completion、cost，单次计时可能波动。缓存读写耗时见各目标 summary.json。', '',
        '各目标 `audit.json` 保存全部场景路线重放及指标核验；`paired_scenarios.csv` 保存逐场景差值。', ''])
    (output / 'REPORT.md').write_text('\n'.join(lines))
    print('\n'.join(lines[6:13]), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=HERE.parent / 'z_test_path/data/n100_obstacles_example.pkl')
    parser.add_argument('--output_dir', type=Path, default=HERE / 'results/n100_obstacles_objectives_run01')
    parser.add_argument('--cpu_threads', type=int, default=1)
    parser.add_argument('--path_workers', type=int, default=1)
    parser.add_argument('--warmup_scenarios', type=int, default=1)
    parser.add_argument('--visibility_epsilon', type=float, default=1e-6)
    parser.add_argument('--cache_dir', type=Path, default=HERE.parent / 'z_test_path/cache')
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--overwrite', action='store_true')
    opts = parser.parse_args()
    output = opts.output_dir.resolve()
    common = ['--data', str(opts.data.resolve()), '--protocol', 'capam_full_visit',
              '--cpu_threads', str(opts.cpu_threads), '--warmup_scenarios', str(opts.warmup_scenarios),
              '--visibility_epsilon', str(opts.visibility_epsilon), '--cache_dir', str(opts.cache_dir.resolve()),
              '--offset', str(opts.offset), '--path_workers', str(opts.path_workers)]
    if opts.num_samples is not None:
        common += ['--num_samples', str(opts.num_samples)]
    if opts.overwrite:
        common += ['--overwrite']
    # Check both destinations before spending time on either objective.
    if not opts.overwrite:
        for objective in ('completion', 'cost'):
            directory = output / objective / 'capam_full_visit'
            if any((directory / name).exists() for name in ('summary.json', 'scenarios.csv', 'groups.csv', 'routes.jsonl')):
                parser.error('Existing evaluation in {}; use a new output_dir or --overwrite'.format(directory))
    env = dict(os.environ, PYTHONHASHSEED='0')
    for objective in ('completion', 'cost'):
        directory = output / objective
        subprocess.run([sys.executable, str(HERE / 'evaluate.py'), *common,
                        '--objective', objective, '--output_dir', str(directory)], env=env, check=True)
        subprocess.run([sys.executable, str(HERE / 'audit_results.py'), '--results', str(directory)], env=env, check=True)
    compare(output)
    print('Comparison: ' + str(output / 'REPORT.md'), flush=True)


if __name__ == '__main__':
    main()
