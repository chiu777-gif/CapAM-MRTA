# BiG-MRTA 已有结果索引

这里区分**主评测结果**与**独立补充分析**。使用方法见 [快速开始](../README.md) 和 [完整中文说明书](../MANUAL.zh-CN.md)。

## 先找主结果

| 数据集 | 场景与机器人数量 | 主结果设置 | 汇总入口 |
|---|---|---|---|
| 原始 TAPTC `data/mrta_gini` | 96 场景，每场景 100 任务；2/3/5/7 个机器人各 24 场景 | CPU 4 线程，batch=1，每协议首场景预热一次 | [taptc96/report.md](taptc96/report.md) |
| 训练同分布 `n100_train_matched_1000_seed20261024.pkl` | 1000 场景，每场景 100 任务；2–7 个机器人 | CPU 4 线程，batch=1，每协议首场景预热一次 | [1000 场景 report.md](n100_train_matched_1000_seed20261024/report.md) |

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
