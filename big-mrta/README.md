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

## 障碍地图：可见图 + Dijkstra

直接评测 `z_test_path` 的障碍数据，无需训练：

```bash
"$PYTHON" big-mrta/evaluate.py \
  --data z_test_path/data/n100_obstacles_example.pkl \
  --protocol both --path_planner visibility_dijkstra \
  --visibility_epsilon 1e-6 --cache_dir z_test_path/cache \
  --path_workers 4 \
  --cpu_threads 1 --warmup_scenarios 1 \
  --output_dir big-mrta/results/n100_obstacles_visibility_run02
```

默认 `--path_planner auto`：带障碍地图时自动使用可见图 + Dijkstra，其余数据保持欧几里得距离。先为所有选中场景预计算任务间、机器人起点到任务的路径，然后两种协议的选择策略、移动时间、cost 和重放均查距离表。缓存格式和键与 `z_test_path` 相同，可直接复用；地图、坐标、有效机器人、规划器源码或 epsilon 改变时会使用不同缓存。

`--path_workers 4` 使用最多 4 个进程，按场景并行准备路径（含缓存读取）；默认 `1`。实际进程数不超过选中的场景数，无障碍数据为 `0`。每个规划子进程的数值库线程数限制为 1。它独立于 `--cpu_threads`；求解仍逐场景串行进行。

首次未命中时执行规划；再次运行只读取缓存。`--rebuild_cache` 强制重算。默认角点偏移为 `1e-6`，与使用 `0.01` 的其他实验比较时，应显式传 `--visibility_epsilon 0.01`，保持参数一致。机器人为点，路径采用当前闭合障碍碰撞规则。

两个协议分别输出 `summary.json`、`scenarios.csv`、`groups.csv`、`routes.jsonl`，根目录生成 `report.md`。路径规划在两个协议之前只执行一次，两个汇总里的同一份规划耗时不能相加。

## 选择优化目标并对比

新增 `--objective completion|cost`，默认 `completion`：

| 配置 | 决策规则 | 说明 |
|---|---|---|
| `completion` | 保留原始可按时完成边及激励权重；空分配时放宽 deadline 补齐任务 | 完成率启发式，不保证全局最大完成率 |
| `cost` | 对包括逾期任务在内的所有剩余任务建边，Hungarian 最小化本轮新增逾期惩罚 | 局部 cost 匹配，不保证最终总 cost 降低 |

两种目标的公平对比均使用 `capam_full_visit`，访问全部任务，统一统计 cost 与按时完成率。cost 为 `Σ finish/deadline`，仅累加 `finish > deadline` 的任务；按时任务新增惩罚为 0。cost 匹配平局按 SciPy 固定行/任务顺序处理，任务少于机器人且当前机器人未匹配时，选该机器人的最小代价任务。

单独运行 cost：

```bash
"$PYTHON" big-mrta/evaluate.py \
  --data z_test_path/data/n100_obstacles_example.pkl \
  --objective cost --protocol capam_full_visit \
  --cpu_threads 1 --output_dir big-mrta/results/obstacles_cost_run01
```

一条命令依次运行并核验两个目标，输出配对比较报告：

```bash
"$PYTHON" big-mrta/compare_objectives.py \
  --data z_test_path/data/n100_obstacles_example.pkl \
  --cpu_threads 1 --warmup_scenarios 1 --visibility_epsilon 1e-6 \
  --path_workers 4 \
  --output_dir big-mrta/results/n100_obstacles_objectives_run02
```

结果包括 `REPORT.md`、`comparison.json`、`paired_scenarios.csv`，以及 `completion/`、`cost/` 各自完整评测输出和 `audit.json`。比较程序核对数据身份、场景、规划参数、逐场景缓存键、线程和预热；两种目标依次运行，推理时间不含规划及额外核验。

[已有 100 场景对比](results/n100_obstacles_objectives_run01/REPORT.md)中，completion 的平均 cost 为 111.237318、完成率 72.39%；单步 cost 匹配分别为 343.591554、39.80%，效果更差。它会推迟高惩罚任务，当前建议保留默认 completion；不能把这个局部 cost 目标当作最终效果保证。

`--objective cost` 未指定协议时自动选 `capam_full_visit`；显式指定 `upstream_taptc` 或 `both` 会报错，因为原版允许遗漏任务，无法公平计算完整 cost。`completion` 未指定协议时仍选 `both`。两种目标应使用不同输出目录。

## 读结果时看这四项

在对应版本的 `summary.json` 中：

| 字段 | 含义 |
|---|---|
| `completion_percent` | 实际完成时间不晚于截止时间的任务比例，百分数 |
| `mean_cost` | 所有场景完整 cost 的平均值；有不完整解时为 `null` |
| `inference_seconds_per_scenario` | 总推理时间除以场景数，单位为秒；乘 1000 得毫秒 |
| `path_planning_seconds_total` / `path_planning_seconds_per_scenario` | 实际路径规划总耗时 / 每场景平均耗时；缓存命中为 0，缓存读写另计 |

耗时包含完整策略求解及仿真更新，排除加载、路径预计算与缓存读写、预热、额外路线重放和输出写入。缓存读写见 `cache_read_seconds_total`、`cache_write_seconds_total`；整个路径准备阶段墙钟时间见 `path_preprocessing_wall_seconds`，包含进程启动、退出和缓存读写。并行时 `path_planning_seconds_total` 是各场景规划耗时之和，不是并行墙钟时间。`path_workers` / `path_workers_requested` 分别记录实际 / 请求的规划进程数。不同线程设置、不同数据、原版/适配版、单场景延迟与批量均摊时间应分别标注。

安装、单独运行某个版本、完整参数表、目录结构及问题排查见[完整中文说明书](MANUAL.zh-CN.md)。
