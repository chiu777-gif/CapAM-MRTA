# BiG-MRTA 完整中文说明书

本手册说明当前仓库里的实际用法。日常入口是 [README.md](README.md)，已经完成的实验见[结果索引](results/README.md)。

## 目录

1. [用途与目录结构](#1-用途与目录结构)
2. [环境和依赖](#2-环境和依赖)
3. [选择原版或全任务适配版](#3-选择原版或全任务适配版)
4. [运行步骤与命令](#4-运行步骤与命令)
5. [命令行参数](#5-命令行参数)
6. [数据读取与单位](#6-数据读取与单位)
7. [输出文件与统计字段](#7-输出文件与统计字段)
8. [完成率、cost 和时间的解释](#8-完成率cost-和时间的解释)
9. [验证与常见问题](#9-验证与常见问题)
10. [源码来源与复现边界](#10-源码来源与复现边界)

## 1. 用途与目录结构

BiG-MRTA 是无需训练的图匹配方法。评测入口直接读取固定场景，用作者的任务选择函数求解；不接受 CapAM 的 `--model` 或检查点参数。

```text
big-mrta/
├── README.md                   # 快速开始与导航
├── MANUAL.zh-CN.md              # 本说明书
├── evaluate.py                 # 固定数据评测入口
├── baseline.py                 # 两种调度协议和作者策略的衔接
├── audit_results.py            # 检查已导出的路线与指标
├── plot_results.py             # 两种协议的机器人数量分组图
├── test_baseline.py            # 与作者驱动及公共仿真器对照验证
├── requirements.txt            # NetworkX / threadpoolctl 附加依赖
├── upstream/                   # 固定版本的作者原始源码、许可证和哈希
├── .deps/                      # 本目录隔离安装的依赖，已忽略版本管理
├── .cache/                     # Matplotlib 等缓存
└── results/
    ├── README.md               # 已有结果的统一索引
    ├── taptc96/                # 原始 TAPTC 96 场景历史主结果及独立分析
    └── n100_train_matched_1000_seed20261024/
                               # 训练同分布 1000 场景历史主结果
```

本手册中的命令均从项目根目录 `/home/hustcsib/chiu_code/CapAM-MRTA` 运行。上述历史结果目录用于查阅；重新实验使用新的输出目录。

## 2. 环境和依赖

当前机器使用已有 `mrta_c` 环境：

```bash
cd /home/hustcsib/chiu_code/CapAM-MRTA
PYTHON=/home/hustcsib/miniconda3/envs/mrta_c/bin/python
"$PYTHON" -c 'import numpy, scipy, torch, matplotlib; print("基础依赖可用")'
```

`PYTHON` 是当前终端的解释器路径变量。新开终端后重新设置，也可把命令中的 `"$PYTHON"` 替换为解释器绝对路径。

当前目录的附加依赖已准备好。缺少依赖时安装：

```bash
"$PYTHON" -m pip install --target big-mrta/.deps -r big-mrta/requirements.txt
```

固定附加依赖为 `networkx==2.8.8` 和 `threadpoolctl==3.5.0`，脚本会自动把 `.deps` 加入导入路径。`requirements.txt` 只列这两个附加包；NumPy、SciPy、PyTorch、Matplotlib 沿用项目环境。运行验证测试还需要 Pandas。

评测使用 CPU。当前入口没有 GPU、批量大小或场景并行参数；每次逐个场景完整求解，batch size 固定为 1。

## 3. 选择原版或全任务适配版

| 项目 | `upstream_taptc`：原版调度协议 | `capam_full_visit`：全任务适配协议 |
|---|---|---|
| 任务选择函数 | 作者原始 `getNextTask` | 同一个作者函数 |
| 工作耗时 | `ceil(workload/capacity)` | `workload/capacity` |
| 调度 | 作者秒级时间网格；同一时刻先共同决策再更新 | 公共 `StateMRTA` 连续时间、顺序决策 |
| 返回空分配 0 | 按原驱动返回机器人初始位置，可留下未分配任务 | 临时放宽 deadline 后重试，直到所有任务被分配 |
| 仍未匹配到当前机器人 | 沿用原驱动处理 | 该次重试暂时关闭其他机器人的竞争 |
| 完整解 cost | 有未分配任务时为 `null` | 每个任务恰好访问一次，计算有限 cost |
| 完成率 | 实际完成时间 ≤ deadline | 同一公式；放宽约束不意味着算作按时完成 |

用 `--protocol both` 可在同一批数据上分别运行两个协议。文件仍分别存放，不会平均成一个“BiG-MRTA 总结果”。

**需要作者原版行为时，报告 `upstream_taptc`；需要与当前 CapAM 相同公式的完整 cost 时，报告 `capam_full_visit` 并注明适配规则。** 两个协议的完成率和时间都可能不同。全任务适配版不能称为未修改原版的 cost 复现。

## 4. 运行步骤与命令

### 4.1 小样本确认流程

先运行 TAPTC 的前两个场景，检查环境及输出：

```bash
"$PYTHON" big-mrta/evaluate.py \
  --data data/mrta_gini \
  --protocol both --num_samples 2 --offset 0 \
  --cpu_threads 4 --warmup_scenarios 1 \
  --output_dir big-mrta/results/smoke_taptc_cpu4_run01
```

`--num_samples` 取固定读取顺序中的前若干场景，不是随机抽样，也不保证覆盖全部机器人数量。完整评测时省略它。

### 4.2 原始 TAPTC 96 场景

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

这一数据集共 96 个场景，每场景 100 个任务；2、3、5、7 个机器人各 24 个场景。

### 4.3 训练同分布 1000 场景

```bash
"$PYTHON" big-mrta/evaluate.py \
  --data z_test/data/n100_train_matched_1000_seed20261024.pkl \
  --protocol both --cpu_threads 4 --warmup_scenarios 1 \
  --output_dir big-mrta/results/n100_trainmatched_cpu4_run01

"$PYTHON" big-mrta/audit_results.py \
  --results big-mrta/results/n100_trainmatched_cpu4_run01

"$PYTHON" big-mrta/plot_results.py \
  --results big-mrta/results/n100_trainmatched_cpu4_run01
```

每场景 100 个任务。2/3/4/5/6/7 个机器人的场景数分别为 147/172/183/173/149/176。分组使用数据保存的 `n_agents`；没有将同一地图复制后增减机器人。

### 4.4 只跑某个版本

```bash
"$PYTHON" big-mrta/evaluate.py \
  --data data/mrta_gini --protocol upstream_taptc \
  --cpu_threads 4 --warmup_scenarios 1 \
  --output_dir big-mrta/results/taptc96_upstream_cpu4_run01
```

只跑适配版时改成 `--protocol capam_full_visit`，同时使用对应的新输出目录。

`audit_results.py` 和 `plot_results.py` 当前都要求结果根目录下已有**两个协议目录**。只有一个协议的输出时，直接看该协议的 `summary.json`、`scenarios.csv` 和根目录的 `report.md`。需要核验或绘图时，在同一根目录运行另一个协议，保持数据、offset、样本数、线程数和预热设置相同。

### 4.5 与单线程 ILS/EILS 比较

```bash
"$PYTHON" big-mrta/evaluate.py \
  --data data/mrta_gini --protocol both \
  --cpu_threads 1 --warmup_scenarios 1 \
  --output_dir big-mrta/results/taptc96_cpu1_run01
```

程序默认线程数是 4，上述比较命令显式改成 1。线程数只是比较条件之一：本地 ILS/EILS 为优化的 C++ 实现，BiG 保留 Python/NetworkX 实现，适配协议也需注明。已有的[单线程补测](results/taptc96/timing_analysis.md)与历史 4 线程主结果是不同批次的测量。

### 4.6 已有文件与重新运行

输出目录不存在时，程序自动创建；目录中已有当前协议的评测文件时，默认报 `Already exists`。

日常重新实验更换 `run01` 为 `run02` 等目录名。确实要覆盖相同目录的评测文件时，给评测命令加 `--overwrite`。它会重建所选协议输出及根目录报告，不会清空整个目录，也不会重新生成已有的独立补充分析。

`report.md` 是自动生成文件，重新评测会重建；独立分析和长期说明应放在单独的 Markdown 文件中。审计和绘图脚本运行后会重写各自的 `audit.json`、PNG 和 SVG。

## 5. 命令行参数

### `evaluate.py`

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--data PATH` | `z_test/data/n100_train_matched_1000_seed20261024.pkl` | 一个数据文件或 TAPTC 根目录 |
| `--protocol` | `both` | `both`、`upstream_taptc` 或 `capam_full_visit` |
| `--output_dir PATH` | `big-mrta/results/n100_train_matched_1000_seed20261024` | 输出根目录；建议每次显式指定 |
| `--cpu_threads N` | `4` | PyTorch / BLAS 线程设置，正整数；不是并行场景数 |
| `--warmup_scenarios N` | `1` | 对首场景完整预热的次数，每个协议分别执行，不计时 |
| `--offset N` | `0` | 跳过固定读取顺序中的前 N 个场景 |
| `--num_samples N` | 未指定，评测剩余全部 | offset 后最多评测 N 个场景 |
| `--overwrite` | 关闭 | 允许覆盖当前协议的已有评测输出 |
| `--help` | — | 显示实际支持的参数 |

当前 CLI 没有 `--model`、`--device`、`--eval_batch_size`、`--robot_counts` 或 `--seed` 参数。机器人数量来自输入文件；进程自动固定 `PYTHONHASHSEED=0`，稳定作者匹配辅助函数中集合转列表的顺序。这个固定值不是重新生成数据的随机种子。

### `audit_results.py` / `plot_results.py`

两个脚本都只提供 `--results PATH` 与 `--help`；默认 `--results` 为训练同分布 1000 场景的历史结果根目录。运行其他实验时务必传入自己的根目录，**不要传某一个协议子目录**。

```bash
"$PYTHON" big-mrta/evaluate.py --help
"$PYTHON" big-mrta/audit_results.py --help
"$PYTHON" big-mrta/plot_results.py --help
```

## 6. 数据读取与单位

| 输入 | 用法与读取规则 |
|---|---|
| 本项目生成的版本化 `.pkl` 文件 | 保存归一化场景；公共加载器检查版本、字段和尺度，不重新生成数据 |
| `data/mrta_gini` 根目录 | 只读取 `group1/group2/r*.txt` 及其对应 `agent/a*.txt` |
| 单个原始 TAPTC 任务 `.txt` | 可作为 `--data` 输入；需保留其对应的 `agent` 目录结构 |

原始 TAPTC 96 场景入口使用根目录；`group1` 子目录本身不是此 CLI 的目录入口。根目录下的 `n_200`、演示文件、旧转换索引不混入这 96 个场景。任意未标记单位的旧 `.pkl` 也不等同于本项目的版本化文件。

生成文件的坐标和速度同时还原到物理单位，工作量和能力同时还原，截止时间始终保持秒。只使用前 `n_agents` 个有效机器人，不把填充槽当作机器人。

TAPTC 策略矩阵直接读取原始物理文本，速度为 1。这样能避免浮点归一化再还原将整数工作耗时变成略大于整数，进而改变原版的向上取整结果。适配版的公共仿真器仍读取项目标准张量。

版本化输入使用保存的速度及全部场景值。默认尺度为坐标/速度 100、工作量/能力 100；不匹配的版本化数据会报尺度错误。当前公开入口没有修改这些尺度或物理速度的参数。

输入会保存 SHA256：生成数据为文件哈希；TAPTC 为任务和配对机器人文件的哈希清单及清单哈希。历史早期的 1000 场景结果只有 `dataset_sha256`，后来输出另含完整 `dataset_fingerprint`，不影响旧文件的哈希核验。

## 7. 输出文件与统计字段

一次 `--protocol both` 的目录结构：

```text
<output_dir>/
├── report.md                   # 两种协议的机器人分组及总体表格
├── upstream_taptc/
│   ├── summary.json            # 原版汇总和元数据
│   ├── scenarios.csv           # 逐场景结果
│   ├── groups.csv              # 按机器人数量分组
│   └── routes.jsonl            # 逐任务分配与完成时间
└── capam_full_visit/
    ├── summary.json            # 适配版汇总和元数据
    ├── scenarios.csv
    ├── groups.csv
    └── routes.jsonl
```

运行审计后增加 `audit.json`；运行绘图后增加 `metrics_by_robot_count.png` 和 `.svg`。绘图标题当前按每场景 100 个任务生成，与本手册两套实测数据一致。

### `summary.json`：先看这里

| 字段 | 含义 |
|---|---|
| `protocol` | 当前文件属于哪个协议 |
| `scenarios` / `total_tasks` | 场景数 / 总任务数 |
| `completed_tasks` / `completion_percent` | 实际按时完成任务数 / 百分数 |
| `assigned_tasks` / `unassigned_tasks` | 已分配 / 未分配任务数 |
| `late_assigned_tasks` | 已分配但逾期的任务数 |
| `mean_cost` | 平均完整解 cost；有不完整场景时为 `null` |
| `mean_assigned_task_deadline_cost` | 只计算已分配任务的惩罚；不完整时不能代替完整 cost |
| `inference_seconds_total` | 全部实际求解的累计推理时间，秒 |
| `inference_seconds_per_scenario` | 平均每场景完整求解时间，秒 |
| `inference_milliseconds_p50` / `inference_milliseconds_p95` | 场景延迟的中位数 / 95 分位，毫秒 |
| `policy_seconds_per_scenario` | 作者 `getNextTask` 调用累计耗时的每场景平均值，属于完整时间的子集 |
| `evaluation_wall_seconds` | 评测循环总墙钟时间，还包括整理、验证、输出等开销 |
| `relaxed_decisions` | 原策略空分配后，放宽 deadline 重试的决策数量 |
| `by_robot_count` | 按保存的机器人数量分组的同类指标 |
| `cpu_threads` / `warmup_scenarios` | 线程 / 每协议首场景预热次数 |
| `upstream` / `data_sources` / `dataset_sha256` | 作者版本、输入来源、数据身份 |

### CSV 与路线

`scenarios.csv` 对应一场景一行，适合查某个场景的 cost、实际完成数、推理耗时和重试次数。`groups.csv` 是按机器人数量汇总的结果，不是单机器人自己的路线统计。

`routes.jsonl` 每行一个场景。事件包含 `robot`、`task`、`decision_time`、`finish_time` 和 `deadline_relaxed`。机器人从 0 编号，任务从 1 编号；原版未分配任务的完成时间为 `null`。`deadline_relaxed=true` 只说明分配时放宽过约束，统计完成率时仍检查原 deadline。

`upstream_travel_cost` 是作者脚本中的行程累计字段，与本项目的 deadline cost 不同。

## 8. 完成率、cost 和时间的解释

### 完成率

```text
按时完成率 = 实际完成时间 <= 原 deadline 的任务数 / 全部任务数
```

完成时间等于 deadline 也算按时。原作者驱动中的 `numTaskDone` 实际计已分配任务；本评测检查真正的完成时间，不直接使用该计数。

总体完成率按任务数加权。两套实测数据都是每场景 100 个任务，所以也等于场景完成率的平均值。

### Cost

```text
cost = Σ_{finish_i > deadline_i} finish_i / deadline_i
     = 逾期任务数 + Σ_{逾期任务} (finish_i - deadline_i) / deadline_i
```

`mean_cost` 是场景 cost 的平均值。cost 同时反映逾期数量和程度，不能单凭它判断两个数据集哪个更难；较低完成率也可能对应较低 cost。原始 TAPTC 与训练同分布数据的实际分解见[核验说明](results/taptc96/dataset_comparison.md)。

原版允许不分配任务，缺少这些任务的完成时间，所以完整 cost 保留为 `null`，不当作 0。全任务适配版的有限 cost 还取决于补齐策略；不能称为论文未修改 BiG-MRTA 的 cost。

### 推理时间

```text
平均每场景秒数 = inference_seconds_total / scenarios
平均每场景毫秒数 = inference_seconds_per_scenario * 1000
```

计时包含矩阵/状态构造、全部建图与匹配、重试、仿真状态更新；排除文件加载、输入转换、预热、额外重放核验、结果写入。CPU 场景串行处理，batch size=1，因此这里是单场景完整求解延迟。

比较时记录数据集、协议、线程数、预热和实现语言。历史 4 线程主结果、单线程补测、cProfile 剖析是不同记录；剖析会增加程序开销，其秒数不替代普通推理时间。现有 CapAM batch=32 的每场景均摊时间也不等同这里的单场景延迟。

历史 `taptc96/timing_benchmark.json` 是独立单线程补测，`timing_profile.json` 是独立剖析；当前评测、审计和绘图脚本不会自动重建它们。用 `--cpu_threads 1` 可重新得到 BiG 的单线程评测结果，但不会顺带重跑 EILS 或生成四方法比较记录。

## 9. 验证与常见问题

执行代码验证：

```bash
PYTHONHASHSEED=0 "$PYTHON" -m unittest discover -s big-mrta -p 'test_*.py' -v
```

验证包括：作者原始驱动对照、原始 TAPTC 文本精度、单位与有效机器人、无可行任务、任务少于机器人、deadline 相等、源码和输入哈希。

适配版评测时还会对每个结果执行公共 `MRTA.replay`，核对完成时间、cost 和完成数，并检查独立物理时间计算。`audit_results.py` 重新检查导出路线、指标、汇总及数据哈希；它不会重新测量推理时间。

| 现象 | 原因及处理 |
|---|---|
| `python: command not found` | 使用本手册的解释器绝对路径，或先设置 `PYTHON` 变量 |
| `ModuleNotFoundError: networkx` / `threadpoolctl` | 用同一解释器将 `requirements.txt` 安装到 `big-mrta/.deps` |
| `Already exists` | 更换输出目录；确实需要替换时使用 `--overwrite` |
| 审计或绘图找不到某协议的 `summary.json` | 当前根目录只有单协议输出，或 `--results` 错传成子目录；先补齐另一协议 |
| 绘图报 `Incompatible protocol summaries` | 两协议的输入、offset、样本数、机器人分布或线程不一致，使用一致参数重新评测 |
| 原版 `mean_cost` 为 `null` | 存在未分配任务；这是明确的统计规则，不是漏算成零 |
| TAPTC cost 比训练同分布数据小 | 先看完成率、逾期数量和相对超时程度，并注明适配协议 |
| BiG 与 EILS 的速度排序不同于论文 | 当前 EILS 为 C++ 重实现，BiG 保留 Python/NetworkX；还需区分原版和适配版，见计时核验 |
| 与旧结果时间不完全一致 | 单次测量有系统负载与预热波动；先核对数据身份、协议、线程和计时定义 |
| 想指定 4 或 6 个机器人测试 TAPTC96 | 该数据集没有这两组；此入口读取已有 `n_agents`，不替换机器人数量 |
| 输入 TAPTC 的 `group1` 目录时报找不到场景 | 传包含 `group1/group2/agent` 的根目录，或单个任务 `.txt` |

## 10. 源码来源与复现边界

- [CapAM 论文](https://arxiv.org/abs/2205.03321)：IV-A.2 中的 BiG-MRTA 基线，对应参考文献 [16] 的 MRS 2019 工作。
- [作者开源仓库](https://github.com/adamslab-ub/BiG-MRTA)：使用专门针对 TAPTC 的 `bigmrta_taptc.py` 和 `main_bigmrta_TAPTC.py`。
- 固定提交为 `e49cd948653b15a70348d85a65b1b42c847f8b5b`。两份源码、作者 README 和 MIT LICENSE 保存在 `upstream/`，SHA256 见 [manifest.json](upstream/manifest.json)，评测会核对源码是否改动。

默认算法参数沿用作者 TAPTC 驱动：激励模型 6、完整任务状态、Hungarian 匹配、`Q=100`、`Range=1e6`、`MaxCost=1000`、`TotalTime=int(1.1*max(deadline))`。这里的 Q 为任务载荷计数，与输入的工作能力速率不是同一个量。

```text
weight(robot, task)
    = remaining_range_after_travel * exp(-predicted_finish / TotalTime)
```

两个协议保留作者代码的以下行为：驱动最终保留最后一个机器人的距离矩阵，用于所有机器人的激励/航程计算；匹配辅助函数把缺失边补为 0，因此部分返回分配可能逾期。本评测分别记录分配数与真正的按时完成数。

`upstream_taptc` 省略作者驱动中的绘图、文件输出和不断增长的历史数组，保留影响策略的决策与状态更新顺序；数据接入和真正按时完成数的统计在本目录完成。原始 TAPTC 保留文本物理值；生成数据采用其保存的参数。

作者策略的开源复用，不等于复现论文机器上的固定耗时；全任务适配，也不等于作者未修改协议。结果中应同时注明版本、数据和计时条件。
