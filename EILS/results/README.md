# 实验结果索引

日常使用见 [中文说明书](../README.zh-CN.md)。本页说明已有结果的身份与用途；算法代码、数据文件和现有结果保持原样。

## 正式实验

| 目录 | 后端 | 原始场景数 | 评估模式 | 实际求解数 | 从这里阅读 |
|---|---|---:|---|---:|---|
| `taptc96_python` | Python/NumPy | 96 | group1 用 tight，group2 用 slack | 192 | [报告](taptc96_python/REPORT.md)、[完整统计](taptc96_python/summary.json) |
| `taptc96` | C++ | 96 | 每场景运行 tight、slack 两套参数 | 384 | [报告](taptc96/REPORT.md)、[完整统计](taptc96/summary.json) |
| `n100_train_matched_1000` | C++ | 1000 | 每场景运行 tight、slack 两套参数 | 4000 | [报告](n100_train_matched_1000/REPORT.md)、[完整统计](n100_train_matched_1000/summary.json) |

正式实验均运行 ILS 和 EILS，使用完整原文参数、连续服务时间、每场景/配置重复一次。算法优化按时完成数，cost 使用统一的剩余任务补全规则。

目前**没有 Python 后端的 1000 场景全量结果**。不要把 `n100_train_matched_1000` 当成 Python 实验，也不要因当前默认后端改变而重新解释历史耗时。

## TAPTC 96 场景总体结果

下表按原始输入分组汇总：group1 取 tight，group2 取 slack。C++ 从 384 个固定配置记录提取 192 个对应记录；Python 直接运行这 192 个配置。

| 算法 | 按时完成率 | 平均 cost | Python 秒/场景 | C++ 秒/场景 | Python 96 场景总秒数 | C++ 96 场景总秒数 |
|---|---:|---:|---:|---:|---:|---:|
| ILS | 59.729% | 108.978762 | 1.771309 | 0.055017 | 170.045662 | 5.281639 |
| EILS | 60.292% | 108.576147 | 6.127080 | 0.148297 | 588.199655 | 14.236482 |

192 个对应解的路线、完成时间、完成数、cost 和迭代数全部一致。[后端对照报告](taptc96_python/BACKEND_COMPARISON.md)、[机器可读核对结果](taptc96_python/backend_comparison.json)。

时间为每场景完整求解的平均值，两种后端来自独立运行，排除数据读取、预热、重放校验和输出。多配置评估循环总墙钟时间不等于某个算法的推理时间。当前结果不构成原论文速度排序的严格复现。

## 训练分布匹配的 1000 场景

tight/slack 是两套参数，数据没有 TAPTC 的 group1/group2 分组。每个配置都使用相同的全部 1000 个场景。

| 算法 | 参数组 | 后端 | 按时完成率 | 平均 cost | 秒/场景 | 1000 场景总秒数 |
|---|---|---|---:|---:|---:|---:|
| ILS | tight | C++ | 81.304% | 88.786946 | 0.072380 | 72.380199 |
| ILS | slack | C++ | 81.220% | 89.205894 | 0.058860 | 58.859796 |
| EILS | tight | C++ | 82.615% | 83.953691 | 0.273008 | 273.008084 |
| EILS | slack | C++ | 81.652% | 87.082404 | 0.135645 | 135.645165 |

机器人数量分组见 [原始报告](n100_train_matched_1000/REPORT.md)。正式实验均保留逐场景路线和复算记录。

## 调试与中断归档

这些目录用于实现检查，不应放进正式算法对比表。

| 目录 | 身份 | 不用作正式对比的原因 |
|---|---|---|
| `pilot` | C++ 小样本试跑 | 仅前 5 个随机场景，共 20 个解 |
| `taptc96_pilot` | C++ 入口调试 | 仅前 1 个 TAPTC 场景，并使用 `--limit 1` |
| `taptc96_python_interrupted_hypot_check` | 中断的 Python 诊断 | 距离舍入检查后已中断，无完整正式汇总，见 [STATUS.json](taptc96_python_interrupted_hypot_check/STATUS.json) |

修正后的完整 Python 实验是 `taptc96_python`。中断归档不代表当前实现的正式指标。

## 日志对应关系

日志位于 `EILS/`，用于查看进度与退出信息；最终指标以结果目录为准。

| 日志 | 对应实验 |
|---|---|
| `full_evaluation.log` | `n100_train_matched_1000`，C++ |
| `taptc96_evaluation.log` | `taptc96`，C++ |
| `taptc96_python_evaluation.log` | `taptc96_python`，Python |
| `taptc96_python_interrupted_hypot_check.log` | 中断的诊断归档，不是最终实验 |

## 文件身份与新增实验

- 新实验先读配置中的 `backend`、`profile_mode`、数据指纹、样本数、种子和 `limit_override`，不要只看目录名。
- 本页历史 C++ 实验未记录顶层 `backend` / `profile_mode`，可通过 `implementation` 中的编译信息确认。
- `verification.json` 的 `status: passed` 表示路线与指标复算通过；后端一致性另看 `backend_comparison.json`，两者不等于作者原版代码级复现。
- 新实验使用新目录，例如 `taptc96_python_run01`。不要重命名现有结果来表示另一后端，也不要把中断文件拼接成完整结果。
