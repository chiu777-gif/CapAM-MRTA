# BiG-MRTA evaluation

Same saved scenarios; robot counts are grouped as stored, not resampled.

The exact authors' TAPTC driver permits unassigned tasks. Its complete-visit deadline cost is undefined on incomplete solutions (`null`). `capam_full_visit` is an explicitly modified protocol: StateMRTA continuous service times and sequential decisions, with deadline relaxation only on empty assignments to visit all tasks. Its cost uses the same metric as CapAM.

## upstream_taptc

| Robots | Scenarios | Mean cost | On-time completion | Full-solution ms/scenario |
|---:|---:|---:|---:|---:|
| 2 | 24 | undefined (unassigned tasks) | 31.958% | 45.752 |
| 3 | 24 | undefined (unassigned tasks) | 44.875% | 77.084 |
| 5 | 24 | undefined (unassigned tasks) | 65.375% | 151.577 |
| 7 | 24 | undefined (unassigned tasks) | 82.333% | 229.497 |
| All | 96 | undefined (unassigned tasks) | 56.135% | 125.978 |

Unassigned tasks: 4172; relaxed decisions: 0.

## capam_full_visit

| Robots | Scenarios | Mean cost | On-time completion | Full-solution ms/scenario |
|---:|---:|---:|---:|---:|
| 2 | 24 | 223.350809 | 31.958% | 158.447 |
| 3 | 24 | 142.155187 | 44.875% | 178.824 |
| 5 | 24 | 69.892500 | 65.458% | 217.019 |
| 7 | 24 | 29.399927 | 82.375% | 255.994 |
| All | 96 | 116.199606 | 56.167% | 202.571 |

Unassigned tasks: 0; relaxed decisions: 4179.

Timing: CPU, batch size 1, serial scenarios; includes matrix/state construction, every upstream graph/matching call and simulation. Excludes dataset loading/conversion, warmup, replay checks and output I/O. Existing CapAM batch-32 amortized times are a different timing regime.

## 跨数据集 cost 的解释与核验

TAPTC 完成率更低，但适配版 cost 略小，并不表示 TAPTC 更容易：cost 同时统计逾期任务数量与归一化超时程度。复核两套数据的 1096 条完整路线后，未发现读取、单位或指标计算错误；原始物理单位独立重放的按时/逾期分类差异为 0。

详细分解、相同机器人数量的分组比较及有限 cost 的适配边界见 [dataset_comparison.md](dataset_comparison.md)；机器可读核验结果见 [dataset_comparison_audit.json](dataset_comparison_audit.json)。

## 与 ILS/EILS 的运行时间比较

本地 ILS/EILS 为优化的 C++ 重实现，BiG-MRTA 保留作者 Python/NetworkX 代码；适配版另包含所有任务补全和公共仿真器开销。统一 CPU 单线程补测后，ILS/EILS/BiG 原版/BiG 适配版的平均完整求解耗时分别为 52.531/143.220/117.592/186.433 ms。当前原版 BiG 快于 EILS；较慢的是适配版。详见 [计时核验与剖析](timing_analysis.md)。这些是当前实现的性能，不是论文原实现速度比例的复现。
