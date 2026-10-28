# BiG-MRTA 已有结果索引

这里区分**主评测结果**与**独立补充分析**。使用方法见 [快速开始](../README.md) 和 [完整中文说明书](../MANUAL.zh-CN.md)。

## 先找主结果

| 数据集 | 场景与机器人数量 | 主结果设置 | 汇总入口 |
|---|---|---|---|
| 原始 TAPTC `data/mrta_gini` | 96 场景，每场景 100 任务；2/3/5/7 个机器人各 24 场景 | CPU 4 线程，batch=1，每协议首场景预热一次 | [taptc96/report.md](taptc96/report.md) |
| 训练同分布 `n100_train_matched_1000_seed20261024.pkl` | 1000 场景，每场景 100 任务；2–7 个机器人 | CPU 4 线程，batch=1，每协议首场景预热一次 | [1000 场景 report.md](n100_train_matched_1000_seed20261024/report.md) |

障碍地图的新评测：[n100_obstacles_visibility_run01/report.md](n100_obstacles_visibility_run01/report.md)。输入为 `z_test_path/data/n100_obstacles_example.pkl` 的 100 场景，CPU 单线程、每协议首场景预热一次、可见图 + Dijkstra，`visibility_epsilon=1e-6`。全任务适配版平均 cost 为 111.237318，按时完成率为 72.39%，平均推理 233.907 ms/场景。100/100 场景命中共享缓存，实际规划 0 秒，缓存读取共 2.490217 秒；这是热缓存结果。两个协议的路径准备统计共享，不相加。其他实验若使用 epsilon=0.01，需要用相同参数重新评测后比较。

主评测的数值以**对应协议的 `summary.json`** 为准。原版与适配版不合并，有限 cost 属于全任务适配版。

| 要查看的文件 | TAPTC96 | 训练同分布 1000 场景 |
|---|---|---|
| 原版汇总 | [upstream_taptc/summary.json](taptc96/upstream_taptc/summary.json) | [upstream_taptc/summary.json](n100_train_matched_1000_seed20261024/upstream_taptc/summary.json) |
| 适配版汇总 | [capam_full_visit/summary.json](taptc96/capam_full_visit/summary.json) | [capam_full_visit/summary.json](n100_train_matched_1000_seed20261024/capam_full_visit/summary.json) |
| 原版机器人分组 | [groups.csv](taptc96/upstream_taptc/groups.csv) | [groups.csv](n100_train_matched_1000_seed20261024/upstream_taptc/groups.csv) |
| 适配版机器人分组 | [groups.csv](taptc96/capam_full_visit/groups.csv) | [groups.csv](n100_train_matched_1000_seed20261024/capam_full_visit/groups.csv) |
| 原版逐场景结果 | [scenarios.csv](taptc96/upstream_taptc/scenarios.csv) | [scenarios.csv](n100_train_matched_1000_seed20261024/upstream_taptc/scenarios.csv) |
| 适配版逐场景结果 | [scenarios.csv](taptc96/capam_full_visit/scenarios.csv) | [scenarios.csv](n100_train_matched_1000_seed20261024/capam_full_visit/scenarios.csv) |
| 指标图 | [PNG](taptc96/metrics_by_robot_count.png) / [SVG](taptc96/metrics_by_robot_count.svg) | [PNG](n100_train_matched_1000_seed20261024/metrics_by_robot_count.png) / [SVG](n100_train_matched_1000_seed20261024/metrics_by_robot_count.svg) |
| 导出数据核验 | [audit.json](taptc96/audit.json) | [audit.json](n100_train_matched_1000_seed20261024/audit.json) |

每个协议目录下的 `routes.jsonl` 还保存逐任务机器人分配、时间及 deadline 放宽标记。

## completion / cost 配置对比

[100 场景目标对比报告](n100_obstacles_objectives_run01/REPORT.md)：输入 `z_test_path/data/n100_obstacles_example.pkl`，全任务适配协议，单线程、预热一次、epsilon=1e-6，两个目标均命中 100/100 路径缓存。

| 目标 | 平均 cost | 按时完成率 | 平均推理 ms/场景 |
|---|---:|---:|---:|
| completion（默认原策略） | 111.237318 | 72.39% | 192.120 |
| cost（本轮新增惩罚匹配） | 343.591554 | 39.80% | 63.646 |

本次单步 cost 匹配的最终 cost 更高，完成率更低；局部代价最小不保证最终调度更优。完成率配置继续作为默认。数据集、距离和评价公式相同，全部任务都被访问，两个目标均完成 100 场景重放核验。逐场景差值见 [paired_scenarios.csv](n100_obstacles_objectives_run01/paired_scenarios.csv)，机器可读汇总见 [comparison.json](n100_obstacles_objectives_run01/comparison.json)。

## cpu_threads=1 / 4，path_workers=8

[100 场景实测报告](n100_cpu_threads_path8_run01/REPORT.md)：completion / capam_full_visit，epsilon=1e-6，每次预热一次。按冷 CPU1、冷 CPU4、热 CPU4、热 CPU1 的顺序串行运行，推理取每组两次均值。

| cpu_threads | 推理 ms/场景 | 冷缓存路径准备墙钟 s | 热缓存路径准备墙钟 s |
|---|---:|---:|---:|
| 1 | 225.339 | 18.992 | 1.946 |
| 4 | 218.058 | 19.098 | 1.769 |

4 线程推理均值耗时低约 3.23%，但组内波动更大，不能确认稳定加速；路径规划基本持平。两组规划都为 8 进程、每个子进程 1 条数值线程。四次运行所有路线和指标一致，均通过 100 场景重放核验。热缓存实际规划耗时为 0；上表包含缓存读取和进程启动退出。原始汇总见 [comparison.json](n100_cpu_threads_path8_run01/comparison.json)，复现实验见 [benchmark.py](n100_cpu_threads_path8_run01/benchmark.py)。

## TAPTC96 目录里的补充文件分别是什么？

| 文件 | 用途 | 是否属于主评测的时间 |
|---|---|---|
| [dataset_comparison.md](taptc96/dataset_comparison.md) | 解释两套数据完成率与 cost 排序不同，并核验数据和路线 | 不产生新的主评测计时 |
| [dataset_comparison_audit.json](taptc96/dataset_comparison_audit.json) | 上述数据核验的机器可读记录 | 否 |
| [timing_analysis.md](taptc96/timing_analysis.md) | 解释 BiG 与 ILS/EILS 的实现差异和计时范围 | 同时引用主结果与独立分析，需按章节区分 |
| [timing_benchmark.json](taptc96/timing_benchmark.json) | 同一 CPU、**1 线程**的四方法补测汇总、参数和源码哈希 | 否，是独立补测 |
| [timing_benchmark.csv](taptc96/timing_benchmark.csv) | 单线程补测的逐场景耗时 | 否 |
| [timing_profile.json](taptc96/timing_profile.json) | 8 个代表场景的 cProfile 函数开销分析 | 否，剖析本身会增加耗时 |

`evaluate.py` 生成主评测输出，`audit_results.py` 生成 `audit.json`，`plot_results.py` 生成 PNG/SVG。上表其余文件是已保存的独立分析，当前这三个脚本不会自动重建它们。

## 三条阅读规则

1. 报告完成率或 cost 时注明数据集与 `protocol`。原版有未分配任务时，完整 cost 为 `null`，不能按 0 处理。
2. 报告平均耗时时，使用所选记录的总推理秒数除以场景数；`summary.json` 的平均时间单位为秒，乘 1000 转毫秒。
3. 历史主结果为 4 线程；`timing_benchmark.json` 为 1 线程。它们是不同测量记录，均保留，不能把其中一项时间直接替换进另一套报告。

重新实验使用新的输出目录。根目录 `report.md` 会被评测脚本重建；补充说明单独保存，便于区分原始数值和后续解释。
