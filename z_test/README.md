# MRTA 测试数据生成与模型评估

两个入口复用项目的场景定义、单位换算、模型加载及公共仿真器。命令在项目根目录运行，使用已安装项目依赖的 Python 环境，例如 `conda activate mrta_c`。

- `generate_data.py`：按指定规模、物理参数生成固定测试场景。
- `evaluate.py`：加载已有检查点和测试数据，输出任务按时完成率、cost 与推理时间。
- `test_workflow.py`：验证物理参数、混合规模统计、计时及命令行入口。

## 生成测试数据

下面生成 **1000 个场景，每个场景 100 个任务、5 个机器人**：

```bash
python z_test/generate_data.py \
  --graph_sizes 200 \
  --robot_counts 4 6 10 14 \
  --dataset_size 1000 \
  --map_size 100 \
  --agent_max_speed 1 \
  --deadline_min 50 --deadline_max 600 \
  --workload_min 10 --workload_max 30 \
  --work_capacity_min 1 --work_capacity_max 3 \
  --coordinate_scale 100 --workload_scale 100 \
  --seed 20261024 \
  --filename z_test/data/n200_4_14.pkl
```

生成 `.pkl` 数据文件及同名 `.json` 参数说明。参数均使用物理地图/工作单位，截止时间为秒；文件内部统一换算为模型单位，记录版本、单位、种子和生成参数。评估时保留文件内已保存的速度及场景值。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--graph_sizes`，也可用 `--graph_size` | `100` | 每个场景的任务数，可给多个规模 |
| `--dataset_size` | `100` | **每个任务规模**生成的场景数 |
| `--robot_counts` | `2 3 4 5 6 7` | 每个场景从集合中均匀随机选机器人数量 |
| `--n_agents` | 未指定 | 固定机器人数量，与显式 `--robot_counts` 互斥 |
| `--map_size` | `100` | 正方形地图边长；任务和机器人初始位置均匀抽样 |
| `--agent_max_speed` | `1` | 同一场景内机器人的移动速度，地图单位/秒 |
| `--deadline_min/max` | `50 / 600` | 任务截止时间的均匀抽样范围，秒 |
| `--workload_min/max` | `10 / 30` | 工作量的均匀抽样范围 |
| `--work_capacity_min/max` | `1 / 3` | 机器人工作能力的连续均匀抽样范围 |
| `--coordinate_scale` | `100` | 坐标、速度的缩放系数，应与模型一致 |
| `--workload_scale` | `100` | 工作量、工作能力的缩放系数，应与模型一致 |
| `--seed` | `1234` | 可复现的随机种子 |
| `--output_dir` | `z_test/data` | 未指定 `--filename` 时的输出目录 |
| `--overwrite`，也可用 `-f` | 关闭 | 显式覆盖同名数据及参数说明 |

默认物理参数统一取自 `problems/mrta/config.py`；命令行显式参数覆盖默认值。任务数、机器人数量须为正整数；速度与范围端点须为有限正数，且下限不大于上限。设置 `--workload_min 20 --workload_max 20` 可固定工作量为 20。

多规模示例：生成 50、100、200 任务的数据集，**每个文件 100 个场景，共 300 个场景**：

```bash
python z_test/generate_data.py \
  --graph_sizes 50 100 200 --n_agents 5 --dataset_size 100 \
  --seed 20261024 --output_dir z_test/data/multi_n
```

生成器沿用均匀随机分布。原始 TAPTC 的 tight/slack 数据直接交给下面的评估入口读取。若要生成新的正态或混合截止时间分布，需要另行定义生成规则。

## 评估模型

先指定检查点；其旁边须保留对应的 `args.json`：

评估入口自动从模型配置读取 `encoder` 和 `n_encode_layers`，支持通过 `--encoder am`、`--encoder gcapcn` 或 `--encoder mamba` 训练的模型。旧配置缺少 `encoder` 时默认使用 GCAPCN，原有检查点无需转换；测评命令不需要另传编码器选项。训练示例及 Mamba 的排序、RMSNorm 与参考扫描实现说明见项目根目录的 [中文使用说明书](../README.zh-CN.md)。

```bash
CAPAM_CHECKPOINT=outputs/mrta_100/capam_paper_20261023T154551842127/epoch-10.pt
```

评估刚生成的数据：

```bash
python z_test/evaluate.py \
  --model "$CAPAM_CHECKPOINT" \
  --data z_test/data/n100_m5.pkl \
  --device cpu --cpu_threads 4 \
  --eval_batch_size 32 \
  --output_dir z_test/results/n100_m5_epoch10
```

评估 **原始 TAPTC 96 个场景**，保持原始坐标、工作量、机器人能力及截止时间：

```bash
python z_test/evaluate.py \
  --model "$CAPAM_CHECKPOINT" \
  --data data/mrta_gini \
  --device cpu --cpu_threads 4 \
  --eval_batch_size 1 \
  --output_dir z_test/results/taptc96_epoch10
```

`--data` 也可用 `--datasets`，支持一个或多个文件、生成数据所在目录、TAPTC 根目录、单个 `group1/group2` 目录或任务 `.txt` 文件。混合任务规模时使用 `--eval_batch_size 1`。

其他评估选项：

- `--model` 也接受检查点目录，此时选择编号最大的 `epoch-*.pt`；对比不同数据集时宜固定某个检查点文件。
- `--device auto` 为默认值，有 CUDA 时使用 `cuda:0`，否则使用 CPU；可指定 `cuda:1`，或用 `--no_cuda` 指定 CPU。
- 默认贪心解码。多次采样可用 `--decode_strategy sample --width 16`，选择最低 cost 的候选，完成率也来自该候选。
- `--warmup_batches 1` 默认对首批做一次不计时预热，可设为 0。
- `--num_samples 10 --offset 0` 评估部分场景；省略时评估全部。
- `--agent_max_speed` 仅解释没有保存速度的 TAPTC/旧格式文件；新版生成文件使用各自保存的速度。
- 缩放系数从模型保存配置读取，数据文件尺度不匹配时明确报错。
- `mrta_gini/n_200` 的旧字典可通过其索引 `--data data/mrta_gini/taptc_data_sets_n_200.pkl --data_format legacy_normalized` 读取。
- 已有结果目录中的文件默认拒绝覆盖，可换目录或显式使用 `--overwrite`。

## 输出与统计口径

每次评估输出：

| 文件 | 内容 |
|---|---|
| `summary.json` | 模型、编码器类型与层数、输入、设备、解码设置、总体指标、按机器人数量汇总 |
| `scenarios.csv` | 每个场景的任务数、按时完成数、完成率、cost、所在 batch 和耗时 |
| `batches.csv` | 每批的场景数、任务数及实际推理时间 |
| `groups.csv` | 按任务数、机器人数量分组；TAPTC 另按 tight/slack 与截止时间比例分组 |

**任务按时完成率 = 所有场景中按时完成的任务数 / 所有任务数**，完成时间等于截止时间也计为按时。混合任务数时，这与“各场景完成率的简单平均”不同，后者另记为 `mean_scenario_completion_rate`。

**推理时间**测量完整任务选择序列，包括策略计算、公共仿真器的状态更新和候选选择；CUDA 测量前后同步。排除模型加载、文件读取、输入传输、预热、额外路线重放验证及输出写入。

- `inference_seconds_total`：所有实际评估 batch 的推理时间之和。
- `inference_seconds_per_scenario`：总推理时间除以场景数。
- **batch size 为 1** 时，后者为单场景完整解的平均延迟。
- **batch size 大于 1** 时，后者是批量处理的均摊耗时，可用于吞吐比较；实际整批延迟见 `batches.csv`。
- `evaluation_wall_seconds` 还包含评估循环的数据整理、重放验证等开销，因此通常大于纯推理时间。

比较推理时间时保持设备、线程数、batch size、预热次数和解码方式一致。

## 验证代码

```bash
python -m unittest discover -s z_test -p 'test_*.py' -v
```
