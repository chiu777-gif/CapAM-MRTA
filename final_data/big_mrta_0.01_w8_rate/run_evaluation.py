"""Run all four saved maps with fresh path planning, then audit and summarize."""
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

OUTPUT = Path(__file__).resolve().parent
REPO = OUTPUT.parents[1]
MAPS = ('example', 'four_rooms', 'random_blocks_d10_s20261028', 'open')


def execute(command, log):
    with log.open('w') as handle:
        process = subprocess.Popen(command, cwd=REPO, env=dict(os.environ, PYTHONHASHSEED='0'),
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(line, end='', flush=True)
        if process.wait():
            raise RuntimeError('Failed; see ' + str(log))


def main():
    jobs = []
    for name in MAPS:
        data = REPO / 'final_data/map_data' / ('n100_m2-3-4-5-6-7_' + name + '_seed6732.pkl')
        directory = OUTPUT / name
        if any((directory / 'capam_full_visit' / f).exists()
               for f in ('summary.json', 'scenarios.csv', 'routes.jsonl', 'groups.csv')):
            raise FileExistsError('Refusing to overwrite ' + str(directory))
        command = [sys.executable, str(REPO / 'big-mrta/evaluate.py'), '--data', str(data),
                   '--objective', 'completion', '--protocol', 'capam_full_visit',
                   '--path_planner', 'visibility_dijkstra', '--visibility_epsilon', '0.01',
                   '--path_workers', '8', '--cpu_threads', '4', '--warmup_scenarios', '1',
                   '--cache_dir', str(REPO / 'z_test_path/cache'), '--rebuild_cache',
                   '--offset', '0', '--output_dir', str(directory)]
        jobs.append((name, directory, command))
    config = dict(started_at_utc=datetime.now(timezone.utc).isoformat(),
                  objective='completion', objective_label='rate', protocol='capam_full_visit',
                  path_planner='visibility_dijkstra', visibility_epsilon=.01, path_workers=8,
                  cpu_threads=4, warmup_scenarios=1, offset=0, num_samples=None,
                  rebuild_cache=True, cache_policy='Never read old caches during evaluation; write freshly planned paths. '
                                                       'Post-evaluation audit reads the new caches outside measured inference.',
                  source_sha256={name: hashlib.sha256((REPO / 'big-mrta' / name).read_bytes()).hexdigest()
                                 for name in ('evaluate.py', 'baseline.py', 'navigation.py', 'audit_results.py')},
                  commands={name: command for name, _, command in jobs})
    (OUTPUT / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')
    rows = []
    for name, directory, command in jobs:
        print('\n=== Map: ' + name + ' ===', flush=True)
        execute(command, OUTPUT / (name + '.log'))
        summary = json.loads((directory / 'capam_full_visit/summary.json').read_text())
        assert summary['scenarios'] == 300 and summary['cache_hits'] == 0 and summary['cache_misses'] == 300
        assert summary['path_workers'] == 8 and summary['cpu_threads'] == 4
        assert summary['unassigned_tasks'] == 0
        execute([sys.executable, str(REPO / 'big-mrta/audit_results.py'), '--results', str(directory)],
                OUTPUT / (name + '_audit.log'))
        audit = json.loads((directory / 'audit.json').read_text())
        assert audit['protocols']['capam_full_visit']['scenarios_verified'] == 300
        row = dict(map=name, model='big_mrta_rate', completion_percent=summary['completion_percent'],
                   mean_cost=summary['mean_cost'], inference_seconds_total=summary['inference_seconds_total'],
                   inference_seconds_per_scenario=summary['inference_seconds_per_scenario'],
                   path_planning_seconds_total=summary['path_planning_seconds_total'],
                   path_planning_wall_seconds=summary['path_preprocessing_wall_seconds'],
                   scenarios=summary['scenarios'], cache_hits=summary['cache_hits'], cache_misses=summary['cache_misses'],
                   cache_read_seconds_total=summary['cache_read_seconds_total'],
                   cache_write_seconds_total=summary['cache_write_seconds_total'])
        rows.append(row)
        with (OUTPUT / 'maps_summary.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print('Verified ' + name, flush=True)
    report = ['# BiG-MRTA rate 全地图评测', '',
              '输入：final_data/map_data 下四份固定数据，每份 300 场景、每场景 100 任务，共 1200 场景。', '',
              'Python；objective=completion（rate），protocol=capam_full_visit，CPU 4 线程，'
              '路径 8 进程，visibility_dijkstra，epsilon=0.01，首场景预热一次，offset=0，全量评测。', '',
              '所有场景强制重新规划（--rebuild_cache），测评不读取旧路径缓存；规划后推理查询本次生成的距离表。'
              '新规划写入共享缓存；测评后额外审计读取本次缓存，不计入推理或路径准备时间。', '',
              '| 地图 | 场景数 | 平均 cost | 按时完成率 | 推理 ms/场景 | 路径准备墙钟 s | 规划进程耗时之和 s |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        report.append('| {} | {} | {:.6f} | {:.3f}% | {:.3f} | {:.3f} | {:.3f} |'.format(
            row['map'], row['scenarios'], row['mean_cost'], row['completion_percent'],
            row['inference_seconds_per_scenario'] * 1000, row['path_planning_wall_seconds'], row['path_planning_seconds_total']))
    report.extend(['', '推理计时包括完整策略、匹配与状态更新，排除规划、缓存 I/O、预热、额外重放和结果写入。'
                   '路径准备墙钟包括进程启动退出、实际规划、适配与缓存写入；规划进程耗时之和不能作为并行墙钟时间。', '',
                   '每个场景均访问全部任务，统计原始 deadline 下的完成率和逾期 finish/deadline 惩罚。'
                   'completion 为原始激励匹配启发式，不保证全局最大完成率。', '',
                   '所有地图均通过 300 场景路线重放与指标核验，测评缓存命中 0/1200。'
                   '各地图 capam_full_visit/ 下保存 summary.json、scenarios.csv、groups.csv、routes.jsonl，'
                   '地图目录的 audit.json 保存核验记录，根目录 maps_summary.csv 为汇总。', ''])
    (OUTPUT / 'REPORT.md').write_text('\n'.join(report))
    (OUTPUT / 'verification.json').write_text(json.dumps(dict(
        passed=True, verified_scenarios=1200, cache_hits=0, cache_misses=1200,
        finished_at_utc=datetime.now(timezone.utc).isoformat(), maps=list(MAPS)), indent=2) + '\n')
    print('\nAll 1200 scenarios completed and verified: ' + str(OUTPUT / 'REPORT.md'), flush=True)


if __name__ == '__main__':
    main()
