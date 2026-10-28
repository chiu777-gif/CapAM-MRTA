"""Paired CPU-thread benchmark; run with the project's mrta_c interpreter."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile

OUTPUT = Path(__file__).resolve().parent
REPO = OUTPUT.parents[2]


def run(command, log):
    with log.open('w') as handle:
        process = subprocess.Popen(command, cwd=REPO, env=dict(os.environ, PYTHONHASHSEED='0'),
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(line, end='', flush=True)
        if process.wait():
            raise RuntimeError('Command failed; see ' + str(log))


def main():
    records = []
    # ABBA order: each thread setting has a forced rebuild and a warm-cache run.
    cases = [('cold_cpu1', 1, True), ('cold_cpu4', 4, True),
             ('warm_cpu4', 4, False), ('warm_cpu1', 1, False)]
    with tempfile.TemporaryDirectory(prefix='big-mrta-path8-benchmark-') as cache:
        for name, threads, cold in cases:
            directory = OUTPUT / name
            command = [sys.executable, str(REPO / 'big-mrta/evaluate.py'),
                       '--data', str(REPO / 'z_test_path/data/n100_obstacles_example.pkl'),
                       '--objective', 'completion', '--protocol', 'capam_full_visit',
                       '--cpu_threads', str(threads), '--path_workers', '8',
                       '--warmup_scenarios', '1', '--visibility_epsilon', '1e-6',
                       '--cache_dir', cache, '--output_dir', str(directory)]
            if cold:
                command.append('--rebuild_cache')
            print('\nBenchmark ' + name, flush=True)
            run(command, OUTPUT / (name + '.log'))
            summary = json.loads((directory / 'capam_full_visit/summary.json').read_text())
            assert summary['scenarios'] == 100 and summary['path_workers'] == 8
            assert summary['cache_misses' if cold else 'cache_hits'] == 100
            records.append(dict(run=name, cache_mode='cold' if cold else 'warm', cpu_threads=threads,
                                **{key: summary[key] for key in (
                                    'scenarios', 'path_workers', 'dataset_sha256', 'mean_cost', 'completion_percent',
                                    'inference_seconds_total', 'inference_seconds_per_scenario',
                                    'path_preprocessing_wall_seconds', 'path_planning_seconds_total',
                                    'path_planning_cpu_seconds_total', 'cache_read_seconds_total',
                                    'cache_write_seconds_total', 'cache_hits', 'cache_misses')}))
        # Verify routes and metrics after all timed runs, without recomputing paths.
        expected = None
        for name, _, _ in cases:
            directory = OUTPUT / name
            routes = (directory / 'capam_full_visit/routes.jsonl').read_text()
            if expected is None:
                expected = routes
            assert routes == expected, 'Thread setting changed routes: ' + name
            run([sys.executable, str(REPO / 'big-mrta/audit_results.py'), '--results', str(directory)],
                OUTPUT / (name + '_audit.log'))
    averages = {}
    for threads in (1, 4):
        rows = [row for row in records if row['cpu_threads'] == threads]
        averages[str(threads)] = dict(
            inference_ms_per_scenario=statistics.mean(row['inference_seconds_per_scenario'] * 1000 for row in rows),
            inference_ms_per_scenario_runs=[row['inference_seconds_per_scenario'] * 1000 for row in rows],
            cold_path_preprocessing_wall_seconds=next(row['path_preprocessing_wall_seconds'] for row in rows if row['cache_mode'] == 'cold'),
            warm_path_preprocessing_wall_seconds=next(row['path_preprocessing_wall_seconds'] for row in rows if row['cache_mode'] == 'warm'))
    result = dict(evaluated_at_utc=datetime.now(timezone.utc).isoformat(),
                  method='Serial ABBA order: cold cpu1, cold cpu4, warm cpu4, warm cpu1. '
                         'Each run contains 100 scenarios, completion/full-visit, epsilon=1e-6, path_workers=8, one warmup. '
                         'Cold runs force rebuilding in a private cache; warm runs reuse it. '
                         'Planning subprocesses each use one numerical thread. '
                         'Inference averages cover two runs per setting; no concurrent benchmark runs. '
                         'All routes are identical and independently audited. Private cache deleted after audit.',
                  source_sha256={name: hashlib.sha256((REPO / 'big-mrta' / name).read_bytes()).hexdigest()
                                 for name in ('baseline.py', 'evaluate.py', 'navigation.py')},
                  runs=records, averages=averages,
                  cpu4_relative_to_cpu1={key: averages['4'][key] / averages['1'][key]
                                         for key in ('inference_ms_per_scenario',
                                                     'cold_path_preprocessing_wall_seconds',
                                                     'warm_path_preprocessing_wall_seconds')})
    (OUTPUT / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# BiG-MRTA CPU 线程对比（path_workers=8）', '',
             '数据：`z_test_path/data/n100_obstacles_example.pkl`，100 场景；completion 目标、capam_full_visit 协议、epsilon=1e-6，每次首场景预热一次。', '',
             '四次运行串行执行，顺序为冷缓存 CPU1、冷缓存 CPU4、热缓存 CPU4、热缓存 CPU1。冷缓存强制重算；热缓存全部命中。推理取每组两次均值。', '',
             '| cpu_threads | 推理 ms/场景（两次均值） | 冷缓存路径准备墙钟 s | 热缓存路径准备墙钟 s |',
             '|---:|---:|---:|---:|']
    for threads in ('1', '4'):
        row = averages[threads]
        lines.append('| {} | {:.3f} | {:.3f} | {:.3f} |'.format(
            threads, row['inference_ms_per_scenario'], row['cold_path_preprocessing_wall_seconds'], row['warm_path_preprocessing_wall_seconds']))
    lines.extend(['', '| 运行 | 推理 ms/场景 | 路径准备墙钟 s | 各场景规划耗时之和 s | 缓存命中 |',
                  '|---|---:|---:|---:|---:|'])
    for row in records:
        lines.append('| {} | {:.3f} | {:.3f} | {:.3f} | {}/100 |'.format(
            row['run'], row['inference_seconds_per_scenario'] * 1000, row['path_preprocessing_wall_seconds'],
            row['path_planning_seconds_total'], row['cache_hits']))
    lines.extend(['', '路径准备墙钟包含 8 个规划子进程的启动/退出、适配、缓存读写及实际规划。各场景规划耗时之和是工作进程时间累计，不能当成并行墙钟时间。'
                  '热缓存的实际规划耗时为 0。推理耗时排除路径准备、预热、额外重放与输出写入。', '',
                  '两组路径规划都使用 8 个进程，每个子进程固定 1 条数值线程；cpu_threads 主要改变主进程求解的数值库线程数，不会把每个规划进程改成 4 线程。'
                  '规划墙钟差异反映单次测量波动及进程/I/O 开销，不能据此断言线程参数会加速规划。', '',
                  '四次运行的所有路线与完成时间完全一致，各次 100 场景均通过独立重放核验。平均 cost={:.6f}，完成率={:.3f}%。'.format(
                      records[0]['mean_cost'], records[0]['completion_percent']), '',
                  '推理每组仅两次、冷/热路径准备每组各一次；这是本机本次实测，不是稳定性或显著性结论。详见 comparison.json、各运行 summary.json 和审计记录。', ''])
    (OUTPUT / 'REPORT.md').write_text('\n'.join(lines))
    print('\n'.join(lines[6:11]), flush=True)


if __name__ == '__main__':
    main()
