# CapAM-MRTA 中文使用说明书

本项目使用胶囊图编码器、注意力解码器与 REINFORCE，学习具有工作量、截止时间和机器人工作能力约束的任务分配策略。

当前版本已分离**随机训练场景生成**与**固定测试场景加载**，统一单位换算、状态更新和代价计算。本文对应重构后的接口；旧版示例中的速度参数、数据格式和部分命令不能直接沿用。

## 1. 最重要的约定：命令行使用原始物理单位

所有公开的环境参数使用论文的原始地图/工作单位，时间统一为秒：

| 项目 | 命令行/原始文件 | 模型内部（默认尺度） |
| --- | --- | --- |
| 地图边长 | `--map_size 100` | 1 |
| 移动速度 | `--agent_max_speed 1` | 0.01 |
| 工作量 | `--workload_min 10 --workload_max 30` | 0.1～0.3 |
| 工作能力 | `--work_capacity_min 1 --work_capacity_max 3` | 0.01～0.03 |
| 截止时间 | `--deadline_min 50 --deadline_max 600` | 50～600 秒，不缩放 |

**现在应传 `--agent_max_speed 1`。** 代码在唯一的单位转换函数中将速度与坐标同时除以 `coordinate_scale=100`，模型实际使用 0.01。若仍传旧示例中的 `0.01`，模型内部将得到 0.0001，物理速度会慢 100 倍。

```text
移动时间：原始距离 50 / 原始速度 1 = 模型距离 0.5 / 模型速度 0.01 = 50 秒
工作时间：原始工作量 20 / 原始能力 2 = 模型工作量 0.2 / 模型能力 0.02 = 10 秒
```

工作量和能力一起除以 `workload_scale=100`，保证工作时间不变。只有物理数据进入系统时做一次转换；新格式数据已经归一化，加载时不会再缩放，且会检查文件与模型的尺度一致。

## 2. 项目结构与职责

| 文件 | 职责 |
| --- | --- |
| `problems/mrta/config.py` | 环境参数、默认值和单位尺度的唯一来源 |
| `problems/mrta/data.py` | 随机训练数据、固定数据读取、单位转换、机器人补齐、数据保存 |
| `problems/mrta/state_mrta.py` | 公共事件驱动仿真，计算移动时间和工作时间 |
| `problems/mrta/metrics.py` | 公共超时代价和成功任务统计 |
| `problems/mrta/problem_mrta.py` | 训练数据/测试数据的独立接口及路线重放 |
| `options.py`、`run.py`、`train.py` | 训练配置、训练与验证、检查点恢复 |
| `reinforce_baselines.py` | rollout、指数移动平均基线和预热 |
| `eval.py` | 只加载固定数据与模型，输出评估结果 |
| `generate_mrta_data.py` | 生成与随机训练分布一致的新格式固定场景 |
| `scripts/train_nohup.sh` | 后台训练，终端输出写入 clog，绘图指标写入 mlog |
| `utils/metrics_logger.py`、`plot_metrics.py` | CSV 指标记录与训练/验证曲线绘图 |
| `tests/test_mrta_pipeline.py` | 单位、状态、数据隔离、采样、恢复训练的回归检查 |
| `tests/test_metrics_logging.py` | CSV 加权汇总、后台启动、恢复日志与绘图检查 |

训练集、随机验证集和 rollout 基线数据集共用同一个 `MRTAConfig`，但使用相互独立的随机种子流。它们均采用均匀分布，不包含 tight/slack 分组公式。生成器按需生成场景，不会在训练开始时将 50 万个场景全部放入内存。

固定测试输入单独加载：保留其中的坐标、工作量、机器人能力和截止时间，不会用训练分布替换它们。随机训练场景也不会从测试目录取数据。使用 `--val_dataset` 时，该文件只用于验证，不会用于训练或替换 rollout 的内部数据集。

## 3. 环境

可以继续使用已经验证过的环境：

```bash
conda activate mrta_c
python -c "import torch, numpy, scipy, tqdm; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

验证环境为 Python 3.8.20、PyTorch 1.12.0+cu113、NumPy 1.24.4、SciPy 1.10.1。核心命令不依赖 pandas、Matplotlib、torchvision 或 tensorboard_logger。原始 `environment.yml` 包含旧 CUDA 依赖，已有 `mrta_c` 时无需重建。

以下命令均在项目根目录、Bash 中执行。行尾 `\` 用于续行；布尔开关只写名称，不附加 `True/False`。

## 4. 快速开始

### 4.1 小场景训练，确认流程

现在已经取消“任务数至少为 100”的限制，可以使用 10 个任务调试：

```bash
python run.py \
  --graph_size 10 \
  --n_agents 2 \
  --batch_size 2 \
  --epoch_size 4 \
  --val_size 4 \
  --eval_batch_size 2 \
  --n_epochs 1 \
  --log_step 1 \
  --run_name quickstart \
  --no_cuda
```

这是流程检查配置，不是论文效果复现配置。每轮训练后均执行验证。输出目录类似：

```text
outputs/mrta_10/quickstart_<时间戳>/
├── args.json          # 完整配置与数据/模型版本
├── convergence.csv    # rollout 基线比较记录，其他基线不产生此文件
└── epoch-0.pt         # 模型、优化器、调度器、基线与随机数状态
```

训练和推理共用公式：`移动耗时=距离/速度`、`工作耗时=工作量/能力`。完成时间等于截止时间时计为成功，与“仅严格超时才产生代价”的定义保持一致。

```text
cost = Σ [完成时间 / 截止时间 × I(完成时间 > 截止时间)]
```

代价越低越好；它不是总路程。完成率为按时完成的任务数除以任务总数，越高越好。

### 4.2 论文主实验的训练规模

默认环境分布已经对应论文描述的训练范围。下面把论文表 I 的 500,000 个训练样本映射为当前实现中每轮的 `epoch_size`：

```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
  --graph_size 100 \
  --robot_counts 2 3 4 5 6 7 \
  --agent_max_speed 1 \
  --deadline_min 50 --deadline_max 600 \
  --workload_min 10 --workload_max 30 \
  --work_capacity_min 1 --work_capacity_max 3 \
  --batch_size 500 \
  --epoch_size 500000 \
  --val_size 10000 \
  --eval_batch_size 500 \
  --n_epochs 100 \
  --lr_model 1e-4 \
  --baseline rollout \
  --run_name capam_paper
```

该配置每轮 1000 次更新，共训练 100 轮。训练仍需要相当的计算时间；本文没有执行这项长训练，也不保证数值结果与论文相同。`eval_batch_size=500` 是吞吐选择，并非论文明确指定的评估批次。

论文训练与测试的描述见 [原文第 IV-A 节和表 I](https://arxiv.org/html/2205.03321v1#S4)。要比较论文指标，还须使用相同测试场景、解码方式和模型选择规则。

训练支持通过 `--encoder gcapcn`、`--encoder am` 或 `--encoder mamba` 选择编码器，大小写均可；省略时沿用 GCAPCN。GCAPCN 固定为单层、`K=2, P=3`；AM 使用 `GraphAttentionEncoder`，默认 3 层、8 个注意力头、128 维嵌入和 512 维前馈隐藏层，可用 `--n_encode_layers` 调整 AM/Mamba 层数。三者沿用相同的任务输入特征（坐标、归一化截止时间）、MRTA 解码器上下文、仿真器与代价函数。AM 将单独嵌入的 depot 与任务一起送入自注意力层。

例如，训练 200 任务、4/6/10/14 机器人规模的 AM-RL：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_nohup.sh \
  --encoder am \
  --graph_size 200 --robot_counts 4 6 10 14 \
  --batch_size 500 --epoch_size 500000 \
  --val_size 10000 --eval_batch_size 500 \
  --n_epochs 100 --run_name am_rl_200 \
  --metrics_step 1 --log_step 50
```

将 `--encoder am` 改为 `--encoder gcapcn` 即训练 GCAPCN；AM 的 3 层与 GCAPCN 的 1 层分别是各自默认结构，并不表示参数量相等。

训练 Mamba 时改用 `--encoder mamba --run_name mamba_rl_200`（替换上例对应参数）。Mamba 默认 3 层、128 维、状态维度 8、扩展倍数 1、卷积核宽度 2，固定使用 RMSNorm；`--normalization` 仅用于 GCAPCN/AM。实现只依赖 PyTorch，不需要安装 `einops` 或 `mamba_ssm`。

Mamba 保留单向递推结构：按 `(x, y, 归一化截止时间)` 对任务做字典序排序，depot 固定在序列开头，编码后恢复原任务索引。输入特征完全相同的任务共享编码表示的均值，避免排序并列引入编号依赖。排序规则是输入约定，不是位置学习参数；没有加入双向扫描。该实现使用顺序扫描，尚未使用官方优化 CUDA 内核，训练和推理速度须在相同设备、批次及解码方式下实测。

### 4.3 nohup 后台运行与两类日志

推荐使用启动脚本，自动分离控制台日志与绘图数据。先激活环境，然后传入普通 `run.py` 参数：

```bash
conda activate mrta_c

CUDA_VISIBLE_DEVICES=0 bash scripts/train_nohup.sh \
  --graph_size 100 \
  --robot_counts 2 3 4 5 6 7 \
  --batch_size 500 --epoch_size 500000 \
  --val_size 10000 --eval_batch_size 500 \
  --n_epochs 100 \
  --run_name capam_paper \
  --metrics_step 1 --log_step 50
```

脚本内部使用 `nohup python -u`，自动关闭进度条、合并标准输出和错误输出、断开标准输入，并打印 PID 和实际日志路径。Python 来自当前已激活环境；也可以通过 `PYTHON_BIN` 显式指定解释器。

```text
clog/<任务名_时间戳_PID>.log       # 完整终端输出，包括异常信息
clog/<任务名_时间戳_PID>.pid       # 后台 Python 进程号
mlog/<任务名_时间戳_PID>/
├── train_batch.csv               # 每个被记录的训练 batch
├── train_epoch.csv               # 每轮训练汇总
├── validation_batch.csv          # 每次验证的 batch 指标
└── validation_epoch.csv          # 每轮验证汇总
```

默认每个 batch 都记录。`--metrics_step 10` 每 10 个 batch 记录一次，`--metrics_step 0` 只保留汇总文件；两者都不会关闭 epoch/评估汇总。`--log_step` 仅控制控制台训练指标的打印频率，不影响 CSV。文件按需产生，未记录 batch 时没有对应 batch CSV。

也可以显式使用原生 nohup：

```bash
conda activate mrta_c
mkdir -p clog mlog
CAPAM_JOB="capam_$(date +%Y%m%dT%H%M%S)_$$"

nohup env CUDA_VISIBLE_DEVICES=0 python -u run.py \
  --graph_size 100 \
  --batch_size 500 --epoch_size 500000 \
  --val_size 10000 --eval_batch_size 500 \
  --n_epochs 100 --run_name "$CAPAM_JOB" \
  --metrics_dir "mlog/$CAPAM_JOB" --metrics_step 1 \
  --no_progress_bar \
  > "clog/$CAPAM_JOB.log" 2>&1 < /dev/null &

echo $! > "clog/$CAPAM_JOB.pid"
```

查看进度、进程和停止任务（替换成启动脚本实际输出的文件名）：

```bash
tail -f clog/实际任务名.log
tail -n 5 mlog/实际任务名/train_epoch.csv
ps -p "$(cat clog/实际任务名.pid)" -o pid,etime,%cpu,%mem,args
kill "$(cat clog/实际任务名.pid)"
```

关闭终端后任务仍在后台运行；`Ctrl+C` 结束 `tail -f` 只会停止查看日志，不会停止训练。`kill` 会终止训练；后续应从已完成保存的检查点恢复。

`CLOG_DIR`、`MLOG_DIR` 可调整启动脚本的两个根目录；`--metrics_dir` 可为单次运行指定绘图目录。直接运行 `python run.py` 时，默认 CSV 保存到 `<运行目录>/metrics/`。恢复训练默认建立新的指标目录，step 延续检查点对应的更新次数，原日志不会被改写。

### 4.4 CSV 字段与绘图

CSV 的主要字段为：

| 字段 | 含义 |
| --- | --- |
| `epoch`、`batch` | 从 0 开始的轮次和批次编号 |
| `step` | 从 1 开始的已完成参数更新次数；验证对应当前训练更新次数 |
| `n_samples` | 本条统计覆盖的场景数 |
| `mean_cost` | 平均超时代价，越低越好 |
| `mean_reward` | **定义为 `-mean_cost`**，越高越好；不是额外计算的环境奖励 |
| `cost_std`、`cost_stderr` | 场景代价的标准差、均值标准误差；单个场景时为 0 |
| `actor_loss`、`total_loss` | 训练损失；不要将其当作环境 cost/reward |
| `negative_log_likelihood` | 训练 batch 的平均负对数似然 |
| `grad_norm`、`clipped_grad_norm` | 裁剪前后的梯度范数 |
| `learning_rate`、`duration_seconds` | 当前学习率和对应阶段耗时 |
| `timestamp_utc`、`elapsed_seconds`、`run` | UTC 时间、当前进程已运行秒数和运行标识 |

epoch/评估平均值按场景数加权，不会因为最后一个 batch 不满而偏差。训练 cost 来自采样策略，验证 cost 来自贪心策略，适合分别绘制两条曲线。每条 CSV 记录写入后立即刷新，可在后台训练期间读取。

在 `mrta_c` 环境中用 Matplotlib 生成 PNG，不需要图形界面：

```bash
python plot_metrics.py --metrics_dir mlog/实际任务名
python plot_metrics.py --metrics_dir mlog/实际任务名 --metric mean_reward
python plot_metrics.py --metrics_dir mlog/实际任务名 --x epoch --output figures/cost.svg
```

默认输出 `mlog/实际任务名/mean_cost.png`，包含训练 batch、训练 epoch 和验证曲线。`--metric mean_reward` 输出 `mean_reward.png`。绘图依赖 Matplotlib；未安装时先运行 `python -m pip install matplotlib`。独立测试输出是场景指标表，不作为训练更新曲线绘制。

不传 `--no_cuda` 时，CUDA 可用则使用 GPU，否则使用 CPU。可见多张 GPU 时主入口使用 `DataParallel`；用 `CUDA_VISIBLE_DEVICES=0` 可以限定单卡。验证覆盖的是 CPU，GPU/多卡性能未验证。

## 5. 固定数据生成与独立评估

### 5.1 生成一个新的固定数据集

生成器使用与训练相同的配置定义和单位转换，输出带版本、单位和生成参数的新格式文件：

```bash
python generate_mrta_data.py \
  --graph_sizes 10 \
  --n_agents 2 \
  --dataset_size 8 \
  --seed 2026 \
  --filename data/custom/uniform_10.pkl
```

已有文件默认拒绝覆盖，需要覆盖时显式添加 `-f`。多个任务规模使用 `--graph_sizes 100 200`，此时不能指定单个 `--filename`，应使用 `--data_dir` 和 `--name`。

### 5.2 选择检查点

```bash
find outputs -name 'epoch-*.pt'
export CAPAM_CHECKPOINT="outputs/mrta_10/quickstart_实际时间戳/epoch-0.pt"
```

替换为实际存在的路径。独立评估也接受整个模型目录，此时选择 epoch 编号最大的检查点；同目录的 `args.json` 必须保留。

### 5.3 贪心评估与 JSON/CSV 导出

```bash
python eval.py \
  --model "$CAPAM_CHECKPOINT" \
  --datasets data/custom/uniform_10.pkl \
  --decode_strategy greedy \
  --eval_batch_size 2 \
  --no_cuda \
  -o results/uniform_10.json
```

将 `-o` 后缀改为 `.csv` 可输出 CSV。默认拒绝覆盖，`-f` 允许覆盖。

独立评估还会自动将 `test_batch.csv`、`test_summary.csv` 写到结果文件同名的 `_metrics` 目录，例如 `results/uniform_10_metrics/`。通过 `--metrics_dir mlog/test_name` 可自定义；`--metrics_step` 控制 batch 记录频率，汇总始终保存。汇总还包含 `completion_rate`、解码策略和采样宽度。

结果包含 `cost`、`tasks_done`、`completion_rate`、`n_agents`、任务选择事件序列 `sequence` 和对应机器人编号 `robot_sequence`。任务编号从 1 开始，机器人编号从 0 开始；每个事件的机器人和任务按两个序列的同一位置对应。

`batch_duration_seconds` 是一个批次的模型推理时间，`amortized_duration_seconds` 是将它除以批次大小后的均摊时间，不能直接当作单机器人实际决策延迟。路线重放校验和文件写入不包含在这个推理时间中。

### 5.4 多次采样

```bash
python eval.py \
  --model "$CAPAM_CHECKPOINT" \
  --datasets data/custom/uniform_10.pkl \
  --decode_strategy sample --width 16 \
  --max_calc_batch_size 32 \
  --eval_batch_size 2 \
  --no_cuda \
  -o results/uniform_10_sample.json
```

每个场景采样 16 条路线，保留代价最低的路线和它对应的成功任务数。大采样数会自动分批计算，最后一批无需整除。`--width 16 64` 可分别评估多个采样数量，此时不允许指定单个 `-o`。

### 5.5 直接评估原始 96 个 TAPTC 场景

```bash
python eval.py \
  --model "$CAPAM_CHECKPOINT" \
  --datasets data/mrta_gini \
  --decode_strategy greedy \
  --eval_batch_size 1 \
  --no_cuda \
  -o results/taptc96.json
```

目录输入会读取 `group1/`、`group2/` 中的任务文本，并自动匹配 `agent/` 中的机器人文本，不需要先运行旧转换脚本。使用正式训练好的 100 任务模型进行论文比较；以上变量仅说明路径用法。

原始任务的 tight/slack 截止时间保持不变。也可以只传一个任务文本，或传多个文件。每个批次允许不同机器人数量，会自动补齐机器人槽位；不同任务数混在同一评估列表时使用 `--eval_batch_size 1`。

## 6. 恢复训练、初始化权重与仅验证

恢复**新格式**检查点：

```bash
python run.py \
  --resume "$CAPAM_CHECKPOINT" \
  --n_epochs 5 \
  --run_name resumed
```

会自动读取原目录配置，恢复优化器、学习率调度器、基线和 Python/NumPy/PyTorch 随机数状态，从保存轮次的下一轮额外训练 5 轮。输出写入新运行目录。修改环境、模型、基线、种子或学习率等受保护设置会报错，防止配置悄悄变化；运行轮数、输出目录等运行控制参数可以显式调整。

CPU 回归测试已检查：连续两轮训练与一轮后恢复一轮的模型参数逐位一致。这不意味着跨 GPU、跨依赖版本也保证完全相同。

只初始化模型权重、开启新实验：

```bash
python run.py \
  --load_path "$CAPAM_CHECKPOINT" \
  --graph_size 100 \
  --run_name initialized
```

`--load_path` **现在只加载模型权重**，不会加载旧优化器或基线。其目录存在 `args.json` 时，自动读取编码器、层数、嵌入维度、归一化和 logits 限幅配置；显式参数仍可覆盖，但结构必须与权重匹配。没有配置文件时须自行提供匹配的结构参数。旧版本检查点不能使用 `--resume`，可用于权重初始化或通过 `eval.py` 评估。

已有配置缺少 `encoder` 字段时自动按 GCAPCN 处理，无须修改原配置或权重。包括 `outputs/mrta_100/capam_paper_20261023T154551842127` 在内的当前格式 GCAPCN 检查点仍可用于恢复训练、初始化和评估，已有 AM 权重也沿用原结构。AM、GCAPCN 与 Mamba 的权重结构不同，切换编码器应重新训练；恢复训练时禁止切换编码器。`eval.py` 和 `z_test/evaluate.py` 根据模型配置自动选择编码器，无须额外传参。

主入口仅验证，不构造或计算 rollout 基线：

```bash
python run.py \
  --eval_only \
  --load_path "$CAPAM_CHECKPOINT" \
  --graph_size 10 \
  --val_dataset data/custom/uniform_10.pkl \
  --val_size 8 \
  --eval_batch_size 2 \
  --no_cuda
```

不指定 `--val_dataset` 则使用独立种子生成随机验证集。需要成功率和路线文件时，使用 `eval.py`。

## 7. 参数说明

### 7.1 共享环境参数

默认值只在 `MRTAConfig` 中定义；训练和数据生成入口复用它们。文件评估只接受需要解释物理文件的速度与两个单位尺度。

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--graph_size` | 100 | 随机训练场景任务数，支持小于 100；不裁剪测试文件中的任务 |
| `--robot_counts` | `2 3 4 5 6 7` | 在给定整数集合中均匀抽取训练机器人数量 |
| `--n_agents` | 不指定 | 固定训练机器人数量，与显式 `--robot_counts` 互斥 |
| `--map_size` | 100 | 物理地图边长，任务和机器人初始坐标在地图内均匀抽样 |
| `--agent_max_speed` | 1 | 原始地图单位/秒；归一化后默认是 0.01 |
| `--deadline_min`、`--deadline_max` | 50、600 | 训练任务截止时间的均匀抽样范围，单位秒 |
| `--workload_min`、`--workload_max` | 10、30 | 训练工作量的均匀抽样范围 |
| `--work_capacity_min`、`--work_capacity_max` | 1、3 | 训练机器人处理速率的均匀抽样范围 |
| `--coordinate_scale` | 100 | 同时除坐标和移动速度，保持移动时间不变 |
| `--workload_scale` | 100 | 同时除工作量和工作能力，保持执行时间不变 |
| `--data_format` | `auto` | 新格式/原始 TAPTC 文本自动识别；无单位标记的旧字典须显式选择格式 |

各范围上下限可以相等，表示固定值。参数必须有限且满足正值约束；机器人数量、任务数量为正整数。

### 7.2 训练入口参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--problem` | mrta | 当前只支持 MRTA |
| `--batch_size` | 5 | 每次参数更新的场景数 |
| `--epoch_size` | 10 | 每轮新生成场景数；更新次数为 `ceil(epoch_size/batch_size)` |
| `--val_size` | 10 | 验证样本上限，也是 rollout 内部随机评估集的大小 |
| `--eval_batch_size` | 5 | 验证和 rollout 计算的批次大小 |
| `--val_dataset` | 不指定 | 可选固定验证文件/目录，只用于验证 |
| `--model` | attention | GCAPCN/AM/Mamba 共用的策略模型入口 |
| `--encoder` | gcapcn | 编码器类型，支持 gcapcn/am/mamba，大小写均可 |
| `--embedding_dim` | 128 | 实际同时传入编码器与解码器，须为 8 的正整数倍 |
| `--n_encode_layers` | GCAPCN: 1；AM/Mamba: 3 | GCAPCN 仅支持 1 层；AM/Mamba 可设任意正整数层数 |
| `--normalization` | batch | GCAPCN/AM 支持 batch/instance；Mamba 固定使用 RMSNorm |
| `--tanh_clipping` | 10 | 解码 logits 的 tanh 限幅，0 关闭 |
| `--lr_model` | 1e-4 | Adam 学习率 |
| `--lr_decay` | 1 | 每轮学习率衰减系数，范围 `(0,1]` |
| `--n_epochs` | 100 | 本次调用额外训练的轮数 |
| `--epoch_start` | 0 | 新实验手动起始轮次；恢复时由检查点决定 |
| `--baseline` | rollout | 支持 rollout、exponential、none |
| `--exp_beta` | 0.8 | 指数移动平均基线/预热基线的平滑系数 |
| `--bl_alpha` | 0.05 | rollout 候选模型更新的单侧配对 t 检验阈值 |
| `--bl_warmup_epochs` | 0 | rollout 基线预热轮数，0 关闭 |
| `--max_grad_norm` | 1 | 梯度 L2 裁剪阈值，0 关闭 |
| `--seed` | 1234 | 模型随机种子；数据按划分与轮次派生独立种子 |
| `--num_workers` | 0 | 训练数据加载子进程数；受限环境保持 0 |
| `--detect_anomaly` | 关闭 | 启用 PyTorch 反向传播异常检测，调试使用 |
| `--checkpoint_epochs` | 1 | 每隔 N 轮及最后一轮保存；0 完全禁用保存 |
| `--load_path` | 不指定 | 仅初始化权重，与 resume 互斥 |
| `--resume` | 不指定 | 恢复新格式完整训练状态与原配置 |
| `--eval_only` | 关闭 | 主入口只验证，不创建基线 |
| `--log_step` | 50 | 每隔多少次参数更新打印训练指标 |
| `--metrics_dir` | `<运行目录>/metrics/` | 独立 CSV 指标目录，后台脚本默认放入 mlog |
| `--metrics_step` | 1 | 每 N 个 batch 记录一次 CSV；0 只记录汇总，不受 log_step 影响 |
| `--run_name` | run | 运行名，自动追加带微秒的时间戳 |
| `--output_dir` | outputs | 运行目录根路径 |
| `--no_cuda` | 关闭 | 强制使用 CPU |
| `--no_progress_bar` | 关闭 | 关闭训练进度条 |

### 7.3 独立评估入口参数

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--model` | 必填 | 检查点文件或模型目录，自动读取模型结构 |
| `--datasets` | 必填 | 一个或多个固定数据文件、原始 TAPTC 目录 |
| `--data_format` | auto | 解释无版本旧字典的单位 |
| `--agent_max_speed` | 模型配置或共享默认值 | 原始 TAPTC/旧字典缺少速度时使用；不会覆盖新格式记录的速度 |
| `--coordinate_scale`、`--workload_scale` | 模型配置或共享默认值 | 原始文件的单位转换；新格式文件必须与模型尺度匹配 |
| `--val_size` | 全部 | 最多评估多少个场景 |
| `--offset` | 0 | 展开全部输入后跳过多少个场景 |
| `--eval_batch_size` | 1 | 推理批次；混合任务规模时使用 1 |
| `--decode_strategy` | greedy | greedy 贪心，sample 多次采样后取最低代价 |
| `--width` | 0 | greedy 要求 0；sample 要求正整数，可给多个值 |
| `--softmax_temperature` | 1 | 正值采样温度，越高分布越平缓 |
| `--max_calc_batch_size` | 10000 | 每次采样计算最多展开多少个场景副本，须 ≥ eval_batch_size |
| `--seed` | 1234 | 采样种子 |
| `--results_dir` | results | 未指定 -o 时使用的输出目录 |
| `--metrics_dir` | `<结果文件名>_metrics/` | 测试 batch 和整体评估指标 CSV 目录 |
| `--metrics_step` | 1 | 测试 batch 指标记录间隔；0 仅保留 test_summary.csv |
| `-o` | 不指定 | JSON/CSV 结果文件；多个 width 时不能指定单个文件 |
| `-f` | 关闭 | 允许覆盖结果 |
| `--no_cuda`、`--no_progress_bar` | 关闭 | 使用 CPU、关闭评估进度条 |

### 7.4 数据生成入口参数

支持第 7.1 节的环境参数，但任务规模使用 `--graph_sizes`（可传多个值），不使用 `--graph_size` 或 `--data_format`。

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--graph_sizes` | 100 | 一个或多个任务规模 |
| `--dataset_size` | 100 | 每种规模生成多少个固定场景 |
| `--filename` | 不指定 | 单个规模的完整输出路径 |
| `--data_dir` | data | 默认输出根目录，在其下建立 mrta 子目录 |
| `--name` | uniform | 默认文件名标识 |
| `--seed` | 1234 | 固定数据生成种子 |
| `-f` | 关闭 | 允许覆盖已有数据文件 |

默认文件名为 `<data_dir>/mrta/<name>_<任务数>_seed<seed>.pkl`。

## 8. 数据格式、兼容性与限制

新数据为带版本的 pickle 字典，包含 `format=capam_mrta`、`version=2`、`units`、`generation`、`seed` 和场景列表 `instances`。每个场景包含规范化后的任务/机器人张量和速度。`offset`、样本数上限对多场景文件与路径索引均实际生效。

旧格式处理：

- 原始 TAPTC `.txt`：自动按物理单位读取，保留全部原始截止时间。
- 旧 `loc_data/robot_data` 字典，已由旧脚本除以 100：使用 `--data_format legacy_normalized`。
- 同名结构但数值仍是物理单位：使用 `--data_format taptc`。
- `.pkl` 文件路径列表：展开后加载目标文件；目标缺失会明确报错，不自动生成替代场景。
- `data/mrta/*_nodes_mrta.pkl` 的旧三元组列表：缺少工作量和机器人属性，明确拒绝加载。

旧字典的评估示例：

```bash
python eval.py \
  --model "$CAPAM_CHECKPOINT" \
  --datasets data/mrta_gini/r11a2i00.pkl \
  --data_format legacy_normalized \
  --agent_max_speed 1 \
  --no_cuda \
  -o results/legacy.json
```

该示例要求目标 `.pkl` 已存在。只有原始文本时，直接传 `data/mrta_gini/group1/r11a2i00.txt` 即可。

旧检查点可按相同权重结构加载；独立评估会为旧模型保留原编码器批次归一化约定和机器人聚合约定，但状态更新与指标始终使用修正后的公共仿真器，因此边界情况的指标可能与旧日志不同。旧模型不能直接用于新格式完整恢复训练。只有可信的项目数据和检查点应使用 pickle/PyTorch 加载。

当前仍限定每个场景一个 depot、每个任务一次由一个机器人完成、同一场景内机器人移动速度相同。不同场景可以具有不同速度。束搜索/压缩访问掩码和多 GPU 评估入口没有实现，不再暴露为可用选项。

移除了旧版未接通的 `--initial_size`、`--n_depot`、`--data_distribution`、`--hidden_dim`、`--lr_critic`、`--log_dir`、`--no_tensorboard`、`--checkpoint_encoder`、`--shrink_size` 等开关。当前不支持动态任务释放、critic 基线或 TensorBoard 写入，不再用无效参数制造“已配置”的假象。

新模型还修正了机器人补齐槽位的偏置贡献，以及编码器中跨场景共享的特征归一化，使贪心评估不会因同批次其他场景或额外补齐机器人而改变结果。保留当前单层 `K=2、P=3` 的胶囊编码结构；没有在本次修改中实现 P 或层数消融。

## 9. 检查与验证

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m unittest discover -s tests -p 'test_*.py' -v
```

检查内容包括：物理单位/模型单位的时间与代价一致；与独立堆调度模拟器对照；不同场景速度；不启用机器人不会被选择；状态不修改输入或历史状态；训练分布与固定测试输入分离；旧单位歧义拒绝；新格式回读与切片；原始 96 个 TAPTC 场景读取；训练、推理与重放的代价一致；采样成功数与最佳路线匹配；批次/补齐不改变新模型贪心结果；检查点恢复；rollout、预热和结果导出。

测试文件和短训练产物写入临时目录。验证不等同于论文长训练收敛或 GPU 性能复现。

日志检查还覆盖了真实 nohup 启动、带空格的日志路径、CSV 实时刷新、末尾小批次的加权汇总、恢复后的 step、控制台频率与指标频率分离，以及 PNG 曲线生成。

论文引用信息见 [原 README](README.md)。
