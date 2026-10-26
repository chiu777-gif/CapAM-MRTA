# BiG-MRTA 使用入口

本目录用于在固定 MRTA 数据上运行 BiG-MRTA，统计**任务按时完成率、cost 和单场景完整求解耗时**。算法不需要训练，也不需要模型检查点。

| 你要做什么 | 从哪里开始 |
|---|---|
| 查看已经跑好的 96 / 1000 场景结果 | [结果索引](results/README.md) |
| 重新评测数据 | 本页“快速运行” |
| 查所有参数、输出字段、数据格式和常见问题 | [完整中文说明书](MANUAL.zh-CN.md) |
| 理解原版与适配版的区别 | 本页“先选评测版本” |
| 查看 cost 倒挂的核验 | [跨数据集指标说明](results/taptc96/dataset_comparison.md) |
| 查看与 ILS/EILS 的速度差异 | [计时核验与剖析](results/taptc96/timing_analysis.md) |

## 先选评测版本

| `--protocol` | 用途 | 完整 cost |
|---|---|---|
| `upstream_taptc` | 直接使用作者策略，保留原始 TAPTC 驱动的调度行为 | 有未分配任务时为 `null` |
| `capam_full_visit` | 使用当前公共仿真器，并补齐全部任务，得到与 CapAM 同公式的 cost | 有限；属于适配版结果 |
| `both`，默认 | 同一批数据分别跑上面两个版本，输出两套结果 | 分别保存，不能混合 |

**论文原始 BiG-MRTA 与全任务适配版是两种评测协议。报告有限 cost 时，要注明 `capam_full_visit`。**

## 快速运行

以下命令均在项目根目录执行。当前机器已有 `mrta_c` 环境及本目录依赖；正常使用直接运行即可。

```bash
cd /home/hustcsib/chiu_code/CapAM-MRTA
PYTHON=/home/hustcsib/miniconda3/envs/mrta_c/bin/python
```

评测**原始 TAPTC 96 场景**，运行两个版本，再核验和绘图：

```bash
"$PYTHON" big-mrta/evaluate.py \
  --data data/mrta_gini \
  --protocol both --cpu_threads 4 --warmup_scenarios 1 \
  --output_dir big-mrta/results/taptc96_cpu4_run01

"$PYTHON" big-mrta/audit_results.py \
  --results big-mrta/results/taptc96_cpu4_run01

"$PYTHON" big-mrta/plot_results.py \
  --results big-mrta/results/taptc96_cpu4_run01
```

然后打开：

```text
big-mrta/results/taptc96_cpu4_run01/report.md
```

评测**训练同分布 1000 场景**时，将上面的评测命令替换为：

```bash
"$PYTHON" big-mrta/evaluate.py \
  --data z_test/data/n100_train_matched_1000_seed20261024.pkl \
  --protocol both --cpu_threads 4 --warmup_scenarios 1 \
  --output_dir big-mrta/results/n100_trainmatched_cpu4_run01
```

后续核验、绘图也要使用这个新的结果目录。再次运行时换成 `run02` 等新目录；已有文件默认拒绝覆盖。

示例显式采用 4 线程，对应历史主结果的设置。比较本地单线程 ILS/EILS 时，改为 `--cpu_threads 1`，并使用包含 `cpu1` 的新结果目录；实现语言和优化程度仍需一并注明。

## 读结果时看这三项

在对应版本的 `summary.json` 中：

| 字段 | 含义 |
|---|---|
| `completion_percent` | 实际完成时间不晚于截止时间的任务比例，百分数 |
| `mean_cost` | 所有场景完整 cost 的平均值；有不完整解时为 `null` |
| `inference_seconds_per_scenario` | 总推理时间除以场景数，单位为秒；乘 1000 得毫秒 |

耗时包含完整策略求解及仿真更新，排除加载、预热、额外路线重放和输出写入。不同线程设置、不同数据、原版/适配版、单场景延迟与批量均摊时间应分别标注。

安装、单独运行某个版本、完整参数表、目录结构及问题排查见[完整中文说明书](MANUAL.zh-CN.md)。
