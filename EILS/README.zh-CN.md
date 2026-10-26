# ILS / EILS 中文使用说明书

本说明书以当前代码和已保存实验为准。日常评估使用 `evaluate.py`，完成后用 `verify_results.py` 复算。默认后端为 Python/NumPy，所有结果保存在 `EILS` 内。

## 1. 先区分概念和参数

| 概念 | 可选值或含义 | 控制什么 |
|---|---|---|
| 算法 | `ils`、`eils` | 搜索中的扰动与停止规则 |
| 后端 | `python`、`cpp` | 用哪种语言实现同一套搜索规则 |
| 参数组 | `tight`、`slack` | 各算法的搜索预算和扰动系数 |
| 数据分组 | TAPTC 的 `group1`、`group2` | 输入场景的真实截止时间分布 |
| 评估模式 | 固定参数、按 TAPTC 分组匹配参数 | 每个场景运行哪些参数组 |
| 服务时间 | `continuous`、`ceil` | 完成任务所花的工作时间如何计算 |

每种算法都能使用任意一套参数。切换 Python/C++ 后端不改变各算法原有的预算；ILS 与 EILS 本身的预算并不相同。

两者都通过可行插入构造路线，再删除部分已分配任务并重新插入。最好解仅在按时完成任务数严格增加时更新；完成数相同则保留先发现的解，不用 cost 决定并列结果。

| 参数组 | ILS 连续无改进上限 | ILS 删除量阈值系数 | EILS 最大构造轮数 | EILS 删除比例 |
|---|---:|---:|---:|---:|
| tight | 600 | 1/2 | 3000 | 1/3 |
| slack | 600 | 2/5 | 5000 | 1/8 |

- ILS 在连续 600 轮无严格改善后停止；改善后计数重置，总轮数可能超过 600。
- EILS 达到 3000 或 5000 次构造后停止。
- 两者在全部任务均能按时完成时提前停止。
- ILS 使用循环变化的删除起点和删除量；EILS 使用随机起点，以及随当前完成数变化的删除量。

参数来源及边界处理见 [SOURCES.md](SOURCES.md)。

## 2. 环境与执行位置

以下命令均在**项目根目录、同一终端**执行，并沿用 `EILS_PY`。这是本机已安装项目依赖的环境；换机器时将变量改为自己的 Python 路径。

```bash
cd /home/hustcsib/chiu_code/CapAM-MRTA
EILS_PY=/home/hustcsib/miniconda3/envs/mrta_c/bin/python
"$EILS_PY" -c 'import numpy, torch; print(numpy.__version__, torch.__version__)'
```

| 后端 | 运行依赖 | 说明 |
|---|---|---|
| `python`，默认 | Python、NumPy、PyTorch | 搜索在 `python_solver.py`；无需编译器，不调用 C++ 求解库，无需 Numba/Cython |
| `cpp` | 上述依赖，以及支持 C++17 的 `g++` | 首次运行自动将 `solver.cpp` 编译到 `build/libeils.so` |

PyTorch 用于项目的数据读取和校验。这些入口使用 CPU，单场景串行求解；没有 `--device`、`--cpu_threads` 或 `--eval_batch_size` 选项，也不加载模型检查点。

## 3. 已有结果应该看哪个

先查看 [结果目录索引](results/README.md)，不需要重新运行实验。

| 需求 | 推荐查看 |
|---|---|
| Python 在原始 TAPTC 96 场景上的结果 | [taptc96_python/REPORT.md](results/taptc96_python/REPORT.md) |
| Python 与 C++ 是否得到了相同解 | [taptc96_python/BACKEND_COMPARISON.md](results/taptc96_python/BACKEND_COMPARISON.md) |
| 历史 C++ 在原始 TAPTC 96 场景上的结果 | [taptc96/REPORT.md](results/taptc96/REPORT.md) |
| 历史 C++ 在均匀随机 1000 场景上的结果 | [n100_train_matched_1000/REPORT.md](results/n100_train_matched_1000/REPORT.md) |

目前尚无 Python 后端在这 1000 个场景上的全量结果。下面的 1000 场景命令用于创建新实验，不能把历史 C++ 耗时标成 Python 耗时。

## 4. 先用小样本确认命令可运行

下面检查前两个 TAPTC 场景，使用小搜索预算，不作为正式性能结果。

```bash
"$EILS_PY" EILS/evaluate.py \
  --backend python \
  --data data/mrta_gini \
  --match_taptc_groups \
  --num_samples 2 \
  --limit 10 \
  --seed 20261025 \
  --output_dir EILS/results/manual_smoke_python_run01

"$EILS_PY" EILS/verify_results.py EILS/results/manual_smoke_python_run01
```

`--limit 10` 对 ILS 表示连续 10 轮无改进上限，对 EILS 表示 10 次构造上限。正式实验应省略该选项，恢复第 1 节的原文参数。

所有示例中的 `run01` 都表示**新实验名称**。输出目录必须位于 `EILS` 内且为空或不存在；再次运行前请换一个未使用的目录名。当前入口没有 `--overwrite`。

## 5. 原始 TAPTC 96 场景

### 推荐：根据原始分组选择参数

```bash
"$EILS_PY" EILS/evaluate.py \
  --backend python \
  --data data/mrta_gini \
  --match_taptc_groups \
  --algorithms ils eils \
  --profiles tight slack \
  --service_time continuous \
  --seed 20261025 \
  --output_dir EILS/results/taptc96_python_run01

"$EILS_PY" EILS/verify_results.py EILS/results/taptc96_python_run01
```

输入为 `group1`、`group2` 的 96 个任务文本，以及对应的 12 个 `agent` 文本。每场景 100 个任务，机器人数量 2、3、5、7 各有 24 个场景。目录中的 `n_200`、演示文件和旧转换 pickle 不参与评估。

`--match_taptc_groups` 根据输入分组选择参数：

| 原始分组 | 场景数 | 使用参数 | 每个场景运行 |
|---|---:|---|---|
| group1：紧截止时间 | 48 | tight | ILS 一次、EILS 一次 |
| group2：宽截止时间 | 48 | slack | ILS 一次、EILS 一次 |

总计 **96 × 2 = 192 次求解**。每种算法在 `summary.json` 的 `matched_models` 下有 96 场景总体结果。在 `models` 下，`ils_tight`、`eils_tight` 等各只有相应的 48 个场景。

完整 96 场景的匹配模式需要保留 `tight slack` 两套参数，以覆盖两类输入。

### 可选：在全部 96 场景上比较两套固定参数

```bash
"$EILS_PY" EILS/evaluate.py \
  --backend python \
  --data data/mrta_gini \
  --algorithms ils eils \
  --profiles tight slack \
  --seed 20261025 \
  --output_dir EILS/results/taptc96_python_fixed_run01
```

这里省略了 `--match_taptc_groups`。每个场景运行 ILS-tight、ILS-slack、EILS-tight、EILS-slack，总计 **96 × 2 × 2 = 384 次求解**。

此时 `models.ils_tight` 等每个配置都有 96 场景；程序还会按输入分组从这批记录提取 `matched_models`，不额外求解，也不根据效果挑选参数。历史 `results/taptc96/` 使用这种运行方式。

## 6. 训练分布匹配的 1000 场景

```bash
"$EILS_PY" EILS/evaluate.py \
  --backend python \
  --data z_test/data/n100_train_matched_1000_seed20261024.pkl \
  --algorithms ils eils \
  --profiles tight slack \
  --service_time continuous \
  --seed 20261025 \
  --output_dir EILS/results/n100_train_matched_1000_python_run01

"$EILS_PY" EILS/verify_results.py EILS/results/n100_train_matched_1000_python_run01
```

文件有 1000 个场景，每场景 100 个任务，机器人数量按保存值为 2–7，各数量的样本数不同。评估不会改机器人数量或重新生成场景。

这里的 `tight/slack` **仅指两套搜索参数**。数据没有原始 TAPTC 的 group1/group2 标签，不能使用 `--match_taptc_groups`。两套参数都在相同的 1000 场景上运行，共 **4000 次求解**；总体指标看 `models`，没有对应的 `matched_models` 总体结果。

只运行一种算法和一套参数可用 `--algorithms eils --profiles tight`，此时共 1000 次求解。Python 全量运行较耗时，先用小样本确认命令，不要直接套用 C++ 的耗时估计。

需要 C++ 后端时，将 `--backend python` 改成 `--backend cpp`，并另设目录，例如 `EILS/results/n100_train_matched_1000_cpp_run01`。

## 7. 全部命令行选项

| 选项 | 默认值 | 作用 |
|---|---|---|
| `--data` | 指定的 `n100_train_matched_1000_seed20261024.pkl` | 一个数据路径；常用输入为版本化 pickle 或 `data/mrta_gini` 原始目录 |
| `--backend` | `python` | `python` 或 `cpp` |
| `--output_dir` | `EILS/results/n100_train_matched_1000_<backend>` | 须为空且在 EILS 内 |
| `--algorithms` | `ils eils` | 一种或两种算法 |
| `--profiles` | `tight slack` | 一套或两套参数 |
| `--match_taptc_groups` | 关闭 | 原始 TAPTC 按输入分组选择一套参数 |
| `--seed` | `20261025` | 非负 64 位整数，用于扰动复现；不会重生成输入数据 |
| `--repeats` | `1` | 每个场景/配置的独立重复次数，须为正整数 |
| `--num_samples` | 全部场景 | 按载入顺序取多少场景；不是随机抽样 |
| `--offset` | `0` | 跳过载入顺序前多少场景 |
| `--service_time` | `continuous` | `continuous` 或 `ceil` |
| `--limit` | 不覆盖 | 调试时覆盖 ILS 无改进上限 / EILS 构造次数上限 |
| `--progress_every` | `25` | 每处理多少输入场景打印进度；`0` 关闭进度输出 |
| `--help` | — | 显示参数列表并退出 |

即使修改 `--data`，默认输出目录仍按上表命名，不会自动改成 TAPTC 名称。**推荐每次显式指定 `--output_dir`。**

匹配模式总次数为 `场景数 × 算法数 × repeats`；固定参数模式再乘参数组数。`--repeats 10` 保留全部重复结果并一起汇总，不挑最好的一次；`scenarios_per_run` 仍表示原始场景数，各模型统计中的 `scenarios` 包含重复次数。

## 8. 输出文件与 JSON 阅读顺序

| 文件 | 建议用途 |
|---|---|
| `REPORT.md` | 首先阅读：后端、数据、总体指标和分组表格 |
| `summary.json` | 完整精度指标、配置、数据/代码指纹与硬件环境 |
| `run_config.json` | 开跑前配置；仅有此文件不代表实验已完成 |
| `scenarios.csv` | 每次求解的场景索引、算法、参数、原始分组、种子、指标、轮数和计时 |
| `groups.csv` | 按算法、参数、机器人数量汇总 |
| `deadline_groups.csv` | TAPTC 按真实截止时间分组汇总；均匀随机数据不生成此文件 |
| `routes.jsonl` | 完整机器人路线、每个任务的完成时间；任务和机器人 ID 从 0 开始 |
| `verification.json` | 执行 `verify_results.py` 后生成的复算结果 |
| `backend_comparison.json`、`BACKEND_COMPARISON.md` | 执行 `compare_backends.py` 后生成的后端核对结果 |

`summary.json` 中：

| 字段 | 含义 |
|---|---|
| `backend` | 新实验记录 `python` 或 `cpp` |
| `profile_mode` | 新实验记录 `fixed` 或 `taptc_matched` |
| `models` | 每个实际算法/参数配置的统计；场景范围随评估模式而变 |
| `matched_models` | 输入分组与参数匹配的总体统计；并非所有实验都有这一结果 |
| `by_robot_count` | 各模型内按机器人数量汇总 |
| `by_deadline_group` | TAPTC 的截止时间分组统计 |
| `completion_percent` | 0–100 范围的完成百分比 |
| `mean_cost` | 每场景 cost 的平均值 |
| `inference_seconds_per_scenario` | 每次完整求解的平均耗时，单位秒 |
| `inference_seconds_total` | 对应统计范围内全部求解耗时之和，单位秒 |
| `evaluation_wall_seconds` | 评估循环总墙钟时间，包含所有实际配置和整理/校验/输出开销 |

读取已有 Python 实验的 96 场景总体指标：

```bash
"$EILS_PY" - <<'PY'
import json
from pathlib import Path

path = Path("EILS/results/taptc96_python/summary.json")
summary = json.loads(path.read_text())
for algorithm, stats in summary["matched_models"].items():
    print(algorithm.upper(),
          "scenarios=", stats["scenarios"],
          "completion_percent=", stats["completion_percent"],
          "mean_cost=", stats["mean_cost"],
          "seconds_per_scenario=", stats["inference_seconds_per_scenario"])
PY
```

历史 `taptc96`、`n100_train_matched_1000` 等实验没有顶层 `backend` / `profile_mode` 字段。这些已知历史实验均为 C++、固定参数模式，可结合 `implementation` 和 [结果索引](results/README.md) 判断，不要因当前默认是 Python 就把旧实验当成 Python 结果。

## 9. 完成率、cost 与补全规则

**按时完成率** = 所有场景按时完成的任务数 / 所有场景任务总数。完成时间等于截止时间也算按时。

单场景 cost：

```text
cost = sum(finish_time[i] / deadline[i]
           for i in tasks if finish_time[i] > deadline[i])
```

`mean_cost` 是单场景 cost 的平均值，不是整个数据集的 cost 总和。使用的是 `finish/deadline`，不是 `(finish-deadline)/deadline`。

原始 ILS/EILS 允许不选择部分任务。为了计算全覆盖 cost，当前实现先冻结最优可行路线，再把未选任务追加给最早空闲的机器人，每次选择旅行时间加服务时间最小的任务。最后所有任务恰好访问一次。

完成率必须与原生搜索得分一致，程序每次都会重放验证；cost 受补全顺序影响。这是明确的适配规则，不是作者公开的原版补全代码。

完成率更高并不保证 cost 更低：cost 同时受逾期任务数量和每个任务的相对逾期程度影响。算法间比较应保持数据、机器人数量、服务时间和补全规则一致。

## 10. 推理时间如何理解

统计的是 CPU 上**单个场景、完整搜索结束并形成完整路线的平均墙钟耗时**，包含距离/服务时间计算、搜索、剩余任务补全和路线提取。

不包含数据读取与转换、Python 模块/动态库加载、C++ 编译、一次预热、独立路线重放校验、指标汇总和文件输出。额外的 `cpu_seconds` 是进程 CPU 时间，二者不要混用。

- 秒换毫秒：`inference_seconds_per_scenario × 1000`。
- 96 场景总时间：看相同模型的 `inference_seconds_total`；不要把多配置合计的 `evaluation_wall_seconds` 当成单个算法的总推理时间。
- `best_iteration` 是最好解所在轮数，不是首次找到最好解的耗时。当前没有记录相应时间，不能用轮数比例倒推。
- 当前时间不等同于找到最终最好解的时间、等质量达成时间或机器人的物理执行时间。
- 与 CapAM 的 GPU/batch-32 均摊时间、BiG-MRTA 的不同协议比较前，应明确设备、线程、任务覆盖规则和计时范围。

本地 ILS 平均运行约 573 轮，EILS 约 3480 轮，Python 耗时约 1.771 秒和 6.127 秒。原论文的 1.342 / 0.992 秒速度顺序尚未被严格复现；后端一致性验证只证明本地两种实现得到了相同解，不证明与作者实现及计时协议完全一致。

## 11. 服务时间和数据单位

`continuous` 使用 `workload/capacity`，是全部现有正式结果的模式，与当前 CapAM 环境一致。`ceil` 对加载后的该比值向上取整，用于离散服务时间选项。

原始文本经项目公共读取器转换为归一化模型单位；版本化 pickle 使用已经保存的单位。坐标与速度、工作量与能力成对缩放，时间仍为秒，不需要手动再乘除 100。

`ceil` 作用于加载后的数值。归一化浮点数在整数边界附近的舍入可能改变上取整结果，不能直接称为“按原始物理文本取整的作者原版结果”。做这类比较时，需要另行核对原始物理值处理路径。

## 12. 复算与 Python/C++ 核对

复算已有正式 Python 结果，不重新搜索：

```bash
"$EILS_PY" EILS/verify_results.py EILS/results/taptc96_python
```

程序重新读取路线，检查数据指纹、任务覆盖、机器人分配、完成时间、cost、完成率、种子、记录完整性和汇总，并写入 `verification.json`。原始数据仍须位于记录路径。

比较已有实验，参数顺序是 **C++ 目录在前，Python 目录在后**：

```bash
"$EILS_PY" EILS/compare_backends.py \
  EILS/results/taptc96 \
  EILS/results/taptc96_python
```

数据指纹、样本范围、算法列表、参数组、种子、服务时间、预算、重复次数和补全规则须一致。Python 的匹配模式可以与 C++ 的固定参数模式比较，只核对相同配置的对应记录。命令会更新 Python 目录下的对照 JSON 和 Markdown，不会重新搜索。

本机已保存的验证结果：192 个对应解无差异，完成时间最大绝对差为 0。时间来自两次独立实验，倍率只描述当时的实现效率。

## 13. 测试、文件职责与常见问题

```bash
"$EILS_PY" -m unittest discover -s EILS -t . -p 'test_*.py' -v
```

本机当前 19 项测试通过。没有 `g++` 时，依赖编译器的跨后端核对测试会跳过，Python 求解仍可使用。完整记录见 [VALIDATION.md](VALIDATION.md)。

| 文件或目录 | 职责 |
|---|---|
| `evaluate.py` | 日常评测入口 |
| `solver.py` | 数据准备、后端选择、完整路线重放与指标校验 |
| `python_solver.py` | Python/NumPy 搜索实现 |
| `solver.cpp` | 保留的 C++ 搜索实现 |
| `data_identity.py` | 输入文件清单与 SHA256 校验 |
| `verify_results.py` | 复算已保存结果 |
| `compare_backends.py` | 比较两种后端的对应结果 |
| `test_*.py`、`test_native.cpp` | 实现验证；日常评估无需直接运行 C++ 测试文件 |
| `SOURCES.md`、`VALIDATION.md` | 文献与适配说明、验证记录 |
| `results/` | 正式实验与调试归档，先看其中的索引 |
| `build/`、`__pycache__/` | 构建、诊断与 Python 缓存，不是实验结果 |
| `*.log` | 运行日志；最终指标以对应结果目录为准 |

| 遇到的情况 | 处理方法 |
|---|---|
| `python` 找不到或缺少 NumPy/PyTorch | 使用第 2 节的 `EILS_PY`，或自己的项目依赖环境 |
| `Output directory is not empty` | 换一个新目录名；不要覆盖正式实验 |
| 输出目录被拒绝 | 将 `--output_dir` 放在 `EILS` 内 |
| 随机 1000 场景使用匹配模式时报错 | 去掉 `--match_taptc_groups`，显式指定参数组 |
| 完整 TAPTC 匹配模式仅指定 tight 后报错 | 保留 `--profiles tight slack`，使 group2 也有参数 |
| 只有部分 CSV/JSONL 和 `run_config.json` | 可能仍在运行或已中断；未完成的记录不能当最终结果 |
| `models.ils_tight.scenarios` 只有 48 | 匹配模式下这是 group1 统计；96 场景总体看 `matched_models.ils` |
| Python/C++ 耗时差几十倍 | 确认后端与预算，不混用两种实现的耗时 |
| 完成率和 cost 同时升高 | 两个指标含义不同，见第 9 节 |
| 想与原论文时间直接对应 | 当前尚未严格复现其运行效率与计时细节，见 `SOURCES.md` |
