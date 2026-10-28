"""Consistent scalar summaries and human-readable evaluation reports."""
from pathlib import Path
import platform

import torch


TIMING_FIELDS = ('inference_seconds_total', 'inference_seconds_per_scenario',
                 'path_planning_seconds_total')


def device_metadata(device, requested, cpu_threads):
    if device.type == 'cuda':
        name, kind = torch.cuda.get_device_name(device), 'GPU'
    else:
        name, kind = platform.processor() or platform.machine(), 'CPU'
        cpuinfo = Path('/proc/cpuinfo')
        if cpuinfo.is_file():
            for line in cpuinfo.read_text().splitlines():
                if line.startswith('model name'):
                    name = line.split(':', 1)[1].strip()
                    break
    return {'device': str(device), 'device_type': kind, 'device_name': name,
            'requested_device': requested, 'cpu_threads': cpu_threads, 'path_device': 'CPU'}


def summarize_rows(rows, metadata):
    count = len(rows)
    total, done = sum(row['task_count'] for row in rows), sum(row['tasks_done'] for row in rows)
    mean_completion = sum(row['completion_rate'] for row in rows) / count
    summary = dict(metadata, scenarios=count, completed_tasks=done, total_tasks=total,
                   completion_rate=done / total, completion_percent=100. * done / total,
                   mean_scenario_completion_rate=mean_completion,
                   mean_scenario_completion_percent=100. * mean_completion,
                   mean_cost=sum(row['cost'] for row in rows) / count,
                   total_distance=sum(row.get('total_distance', 0.) for row in rows),
                   cache_hits=sum(bool(row.get('cache_hit')) for row in rows),
                   cache_misses=sum(not bool(row.get('cache_hit')) for row in rows),
                   inference_batches=len({row['batch_id'] for row in rows}),
                   inference_timing_mode=('single_scenario_latency' if all(row['batch_scenarios'] == 1 for row in rows)
                                          else 'batch_amortized'))
    inference = sum(row['inference_seconds'] for row in rows)
    summary.update(inference_seconds_total=inference,
                   inference_seconds_per_scenario=inference / count,
                   path_planning_seconds_total=sum(row.get('path_planning_seconds', 0.) for row in rows))
    return summary


def timing_definitions():
    return {
        'inference_seconds_total': 'Total encoding, decoding, path-table simulation and candidate-selection '
                                   'time in seconds; excludes planning, warmup, transfers, replay and exports. '
                                   'CUDA is synchronized before and after inference.',
        'inference_seconds_per_scenario': 'Total inference time divided by the number of scenarios, in seconds. '
                                          'Group times use each scenario\'s share of its actual batch time.',
        'path_planning_seconds_total': 'Sum of actual graph construction, path search, optional smoothing '
                                       'and distance-table computation '
                                       'times across scenarios/workers, in seconds; excludes cache I/O. '
                                       'Cache hits contribute zero; parallel worker times are cumulative.',
    }


def write_report(path, summary, groups):
    seconds, count = summary['inference_seconds_total'], summary['scenarios']
    lines = [
        '# 障碍物数据集测评汇总', '',
        '- 检查点：`{}`'.format(summary['checkpoint']),
        '- 编码器：`{}`；数据集：`{}`'.format(summary['encoder'], summary['dataset']),
        '- 路径规划：`{}`（{}）；{}。'.format(
            summary['path_planner'], summary['planner'],
            '栅格细分 {}'.format(summary['path_subdivisions']) if summary['path_planner'] == 'grid_astar'
            else '单位地图拐角偏移 {}'.format(summary['visibility_epsilon'])),
        '- 推理设备：**{} / {}**（{}）；路径规划设备：CPU'.format(
            summary['device_type'], summary['device'], summary['device_name']),
        '- 推理 batch size：{}；CPU 推理线程：{}；路径规划进程：{}'.format(
            summary['eval_batch_size'], summary['cpu_threads'], summary['path_workers']),
        '- 解码：{}，候选数 {}；缓存命中 {}/{}。'.format(
            summary['decode_strategy'], summary['width'], summary['cache_hits'], count), '',
        '## 总体指标', '',
        '| 样本数 | 按时完成任务 / 总任务 | 任务加权完成率 | 场景完成率均值 | 平均 cost | 推理总时间（秒） | 平均每样本推理时间（秒） | 路径规划时间（秒） |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|',
        '| {} | {} / {} | {:.3f}% | {:.3f}% | {:.6f} | {:.6f} | {:.6f} | {:.6f} |'.format(
            count, summary['completed_tasks'], summary['total_tasks'], summary['completion_percent'],
            summary['mean_scenario_completion_percent'], summary['mean_cost'], seconds,
            summary['inference_seconds_per_scenario'], summary['path_planning_seconds_total']), '',
        '## 按机器人数量分组', '',
        '| 机器人数量 | 样本数 | 任务加权完成率 | 场景完成率均值 | 平均 cost | 推理总时间（秒） | 平均每样本推理时间（秒） | 路径规划时间（秒） |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in groups:
        lines.append('| {} | {} | {:.3f}% | {:.3f}% | {:.6f} | {:.6f} | {:.6f} | {:.6f} |'.format(
            row['n_agents'], row['scenarios'], row['completion_percent'], row['mean_scenario_completion_percent'],
            row['mean_cost'], row['inference_seconds_total'], row['inference_seconds_per_scenario'],
            row['path_planning_seconds_total']))
    lines += ['', '## 计时及统计口径', '',
              '- 完成率使用按时完成任务数；同时提供任务加权完成率和各场景完成率的算术平均。平均 cost 为场景 cost 的算术平均。',
              '- 推理时间包含编码、解码、距离查表和候选选择，不包含路径规划、预热、传输、重放和文件输出；GPU 推理计时前后同步 CUDA。',
              '- batch size 大于 1 时，每样本和分组推理时间是按实际 batch 大小分摊的平均值；需要单样本延迟时使用 `--eval_batch_size 1`。',
              '- 路径规划时间为各样本实际建图、路径搜索、平滑（适用时）和距离表计算的累计耗时；命中缓存时为 0，不计缓存读写。并行时为各进程累计时间。', '']
    Path(path).write_text('\n'.join(lines), encoding='utf-8')
