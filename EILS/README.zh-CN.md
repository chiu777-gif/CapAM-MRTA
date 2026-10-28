# ILS / EILS 中文使用说明书

本说明书以当前代码和已保存实验为准。日常评估使用 `evaluate.py`，完成后用 `verify_results.py` 复算。默认后端为 Python/NumPy，实验结果保存在 `EILS` 内；障碍数据的导航缓存默认与 `z_test_path/cache` 共享。

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

以下命令均在**项目根目录、同一终端**执行。先激活已安装项目依赖的 Conda 环境 `mrta_c`，后续使用该环境中的 `python`；新开终端后需重新激活环境。

```bash
cd /home/hustcsib/chiu_code/CapAM-MRTA
conda activate mrta_c
python -c 'import numpy, torch; print(numpy.__version__, torch.__version__)'
```

| 后端 | 运行依赖 | 说明 |
|---|---|---|
| `python`，默认 | Python、NumPy、PyTorch | 搜索在 `python_solver.py`；无需编译器，不调用 C++ 求解库，无需 Numba/Cython |
| `cpp` | 上述依赖，以及支持 C++17 的 `g++` | 首次运行自动将 `solver.cpp` 编译到 `build/libeils.so` |

PyTorch 用于项目的数据读取和校验。这些入口使用 CPU，默认串行求解；可通过 `--workers` 使用多个进程并行处理不同场景，每个进程内仍逐场景求解。没有 `--device`、`--cpu_threads` 或 `--eval_batch_size` 选项，也不加载模型检查点。

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
python EILS/evaluate.py \
  --backend python \
  --data data/mrta_gini \
  --match_taptc_groups \
  --num_samples 2 \
  --limit 10 \
  --seed 20261025 \
  --output_dir EILS/results/manual_smoke_python_run01

python EILS/verify_results.py EILS/results/manual_smoke_python_run01
```

`--limit 10` 对 ILS 表示连续 10 轮无改进上限，对 EILS 表示 10 次构造上限。正式实验应省略该选项，恢复第 1 节的原文参数。

所有示例中的 `run01` 都表示**新实验名称**。输出目录必须位于 `EILS` 内且为空或不存在；再次运行前请换一个未使用的目录名。当前入口没有 `--overwrite`。

## 5. 原始 TAPTC 96 场景

### 推荐：根据原始分组选择参数

```bash
python EILS/evaluate.py \
  --backend python \
  --data data/mrta_gini \
  --match_taptc_groups \
  --algorithms ils eils \
  --profiles tight slack \
  --service_time continuous \
  --seed 20261025 \
  --output_dir EILS/results/taptc96_python_run01

python EILS/verify_results.py EILS/results/taptc96_python_run01
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
python EILS/evaluate.py \
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
python EILS/evaluate.py \
  --backend python \
  --data z_test/data/n100_train_matched_1000_seed20261024.pkl \
  --algorithms ils eils \
  --profiles tight slack \
  --service_time continuous \
  --seed 20261025 \
  --output_dir EILS/results/n100_train_matched_1000_python_run01

python EILS/verify_results.py EILS/results/n100_train_matched_1000_python_run01
```

文件有 1000 个场景，每场景 100 个任务，机器人数量按保存值为 2–7，各数量的样本数不同。评估不会改机器人数量或重新生成场景。

这里的 `tight/slack` **仅指两套搜索参数**。数据没有原始 TAPTC 的 group1/group2 标签，不能使用 `--match_taptc_groups`。两套参数都在相同的 1000 场景上运行，共 **4000 次求解**；总体指标看 `models`，没有对应的 `matched_models` 总体结果。

只运行一种算法和一套参数可用 `--algorithms eils --profiles tight`，此时共 1000 次求解。Python 全量运行较耗时，先用小样本确认命令，不要直接套用 C++ 的耗时估计。

需要 C++ 后端时，将 `--backend python` 改成 `--backend cpp`，并另设目录，例如 `EILS/results/n100_train_matched_1000_cpp_run01`。

### 多进程并行加速

同一终端先执行 `conda activate mrta_c`，再使用下面的 8 进程命令。`--workers` 对 Python 和 C++ 后端都有效，默认值为 `1`。场景之间相互独立，并行时保留原来的输入、随机种子、搜索预算和结果校验。

```bash
python EILS/evaluate.py \
  --backend python \
  --data z_test/data/n100_train_matched_1000_seed20261024.pkl \
  --algorithms ils eils \
  --profiles tight slack \
  --service_time continuous \
  --seed 20261025 \
  --workers 8 \
  --progress_every 1 \
  --output_dir EILS/results/n100_train_matched_1000_python_w8_run01

python EILS/verify_results.py EILS/results/n100_train_matched_1000_python_w8_run01
```

先用 `8` 个进程观察整批耗时和内存占用，再按机器可用 CPU 和内存调整；实际加速比受资源竞争和场景难度差异影响。每个进程会加载 Python/NumPy/PyTorch，因此进程数增加也会增加内存占用。实际进程数不会超过 `场景数 × repeats`。

启动时会立即显示数据路径，随后显示场景数、求解次数和进程数；第一个场景完成后显示进度，之后按 `--progress_every` 指定的场景间隔输出。每个场景的全部配置完成后都会刷新 CSV 和路线文件。并行时场景按完成顺序写入，读取时按 `scenario_index`、`repeat`、算法和参数组识别记录。

已经运行的串行进程不会自动切换为并行。要切换，在原终端按 `Ctrl+C` 结束当前运行，再使用一个新的输出目录启动；旧目录保留部分结果，当前不支持断点续跑。

并行缩短的是**整批评估的墙钟时间**。单次 `inference_seconds` 仍测量一次完整求解的实际延迟，并行时可能受资源竞争影响；`inference_seconds_total` 是各次延迟之和，可以大于 `evaluation_wall_seconds`。比较单进程推理延迟时使用 `--workers 1`，详见第 10 节。

### 障碍数据：可见图 + Dijkstra

`z_test_path/data/n100_obstacles_example.pkl` 有 100 个场景，每个场景 100 个任务。下面只运行 EILS，每个场景分别使用 tight、slack 两套完整搜索预算，共 200 次求解：

```bash
python EILS/evaluate.py \
  --backend python \
  --data z_test_path/data/n100_obstacles_example.pkl \
  --algorithms eils --profiles tight slack \
  --path_planner visibility_dijkstra \
  --visibility_epsilon 1e-6 \
  --cache_dir z_test_path/cache \
  --path_workers 4 --workers 4 \
  --output_dir EILS/results/n100_obstacles_visibility_python_run02

python EILS/verify_results.py EILS/results/n100_obstacles_visibility_python_run02
```

`--path_planner auto` 是默认值：检测到 pickle 中的 `obstacle_map` 就使用可见图 + Dijkstra；普通无障碍数据继续使用欧氏距离。障碍数据不能指定 `euclidean`。Python 和 C++ 后端都支持缓存距离；只运行一套参数可用 `--profiles tight`。

程序先为所选场景预计算任务之间、有效机器人起点到任务的避障距离，再开始任何搜索。插入可行性、扰动后的重建、剩余任务补全，以及最终完成时间、cost 和完成率复算，全部使用这两张距离表。移动时间为 `缓存距离 / max_speed`，服务时间仍按所选模式计算；保留数据保存的坐标、有效机器人数量和归一化单位。

路径规划直接复用 `z_test_path/path_planner.py` 和 `visibility_planner.py`：使用向外偏移的障碍角点建可见图，以欧氏边长加权，再用 Dijkstra 求图上的最短路径。障碍边界也视为碰撞。`visibility_epsilon` 默认 `1e-6`，单位为内部 `[0,1]²` 地图；与 CapAM 等结果比较时，需要使用相同的 epsilon，例如现有 `capam_path_100_d_1` 使用 `0.005`。

同一个场景的规划只执行一次，供所有算法、参数组和 repeats 共享。缓存按地图、坐标、有效机器人数量、地图尺度、规划算法/epsilon 和源码指纹区分，兼容 `z_test_path` 已有的同配置缓存。再次运行须换一个空输出目录，但可继续使用同一个 `--cache_dir`；命中缓存时无需建图或搜索。需要测量重新规划的耗时时增加 `--rebuild_cache`；这会重新计算并更新对应缓存。

`--path_workers` 控制提前规划阶段的进程数；`--workers` 控制后续 ILS/EILS 搜索进程数，两个阶段依次运行。结果看 `REPORT.md` 或 `summary.json`：每套参数的 `models.eils_tight` / `models.eils_slack` 包含平均 cost、完成率、推理总时间及每场景推理时间；路径规划累计时间放在顶层 `path_planning_seconds_total`，供各配置共享，不重复相加。规划命中缓存时该值为 0；缓存读取和规划阶段墙钟时间另有记录。

## 7. 全部命令行选项

| 选项 | 默认值 | 作用 |
|---|---|---|
| `--data` | 指定的 `n100_train_matched_1000_seed20261024.pkl` | 一个数据路径；常用输入为版本化 pickle 或 `data/mrta_gini` 原始目录 |
| `--backend` | `python` | `python` 或 `cpp` |
| `--output_dir` | `EILS/results/n100_train_matched_1000_<backend>` | 须为空且在 EILS 内 |
| `--algorithms` | `ils eils` | 一种或两种算法 |
| `--profiles` | `tight slack` | 一套或两套参数 |
| `--objective` | `completion` | `completion` 最大化按时完成数；`cost` 最小化每轮候选补全全部任务后的 cost，Python/C++ 均支持 |
| `--match_taptc_groups` | 关闭 | 原始 TAPTC 按输入分组选择一套参数 |
| `--seed` | `20261025` | 非负 64 位整数，用于扰动复现；不会重生成输入数据 |
| `--repeats` | `1` | 每个场景/配置的独立重复次数，须为正整数 |
| `--num_samples` | 全部场景 | 按载入顺序取多少场景；不是随机抽样 |
| `--offset` | `0` | 跳过载入顺序前多少场景 |
| `--service_time` | `continuous` | `continuous` 或 `ceil` |
| `--limit` | 不覆盖 | 调试时覆盖 ILS 无改进上限 / EILS 构造次数上限 |
| `--workers` | `1` | CPU 工作进程数，须为正整数；不同场景并行，实际值不超过 `场景数 × repeats` |
| `--path_planner` | `auto` | 障碍数据自动使用 `visibility_dijkstra`，普通数据使用 `euclidean` |
| `--visibility_epsilon` | `1e-6` | 可见图角点向外偏移量，内部 `[0,1]²` 地图单位；与 `z_test_path` 一致 |
| `--cache_dir` | `z_test_path/cache` | 共享的逐场景导航缓存，独立于实验输出目录 |
| `--path_workers` | `1` | 提前规划的 CPU 进程数，最大为所选场景数；普通数据不启动规划进程 |
| `--rebuild_cache` | 关闭 | 重新计算并更新所选场景对应的导航缓存 |
| `--progress_every` | `25` | 第一个场景完成后打印进度，之后每完成指定数量的场景打印一次；`0` 关闭进度输出 |
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
| `path_planning.csv` | 障碍数据每个原始场景的缓存键、命中状态、规划时间和缓存读写时间；不按参数组或 repeats 重复 |
| `path_distances.npz` | 本次搜索使用的距离表快照，带 SHA256；复算时无需共享缓存或重新规划 |
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
| `evaluation_wall_seconds` | 评估循环总墙钟时间，包含所有实际配置和整理/校验/输出开销；并行时也包含进程池启动/关闭及进程间通信 |
| `timing.workers` | 实际 CPU 工作进程数；`workers_requested` 保存命令行请求值 |
| `path_planner` | 本次实际使用的 `euclidean` 或 `visibility_dijkstra` |
| `path_planning_seconds_total` | 全部所选场景实际建图、Dijkstra 搜索和距离表计算的累计秒数；排除缓存读写，命中缓存贡献 0 |
| `path_planning_wall_seconds` | 单独规划阶段的墙钟秒数，包括缓存读写和规划进程启动/通信；不含推理 |
| `cache_hits` / `cache_misses` | 所选原始场景的导航缓存命中/未命中数 |
| `cache_read_seconds_total` / `cache_write_seconds_total` | 导航缓存读写累计秒数，与规划计算、推理分别统计 |

读取已有 Python 实验的 96 场景总体指标：

```bash
python - <<'PY'
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

完成率必须与所选候选的按时完成数一致，程序每次都会重放验证；cost 受补全顺序影响。这是明确的适配规则，不是作者公开的原版补全代码。

### 切换优化目标

默认 `--objective completion` 保留原版按时完成数目标和搜索行为。增加 `--objective cost` 后，每轮先按原有规则构造可行路线，再对全部剩余任务做相同的统一补全，以完整方案的上述 cost 评分，严格更低才更新最好解；同值保留先找到的解，不以完成率作为隐含优先目标。评分完成后恢复可行路线与已用任务标记，再执行原有 ILS/EILS 扰动。最终返回所选候选补全后的完整路线。

cost 模式没有把构造邻域改成任意完整路线的重排邻域，仍使用原有 deadline-feasible 插入与扰动，因此属于 **cost 目标的 ILS/EILS 适配版**，不保证全局最优。ILS 的 `600` 表示对应目标连续无改进次数，改进会重置计数；EILS 的 tight/slack 仍分别为 `3000/5000` 次构造。cost 达到全局下界 `0` 时提前结束。每轮补全评分计入推理时间。

例如对同一数据集运行两种目标（输出目录须各自为空）：

```bash
python EILS/evaluate.py \
  --data z_test_path/data/n100_obstacles_example.pkl \
  --backend python --algorithms ils eils --profiles tight slack \
  --objective cost --seed 6732 \
  --path_planner visibility_dijkstra --visibility_epsilon 0.01 \
  --path_workers 8 --workers 8 \
  --output_dir EILS/results/n100_obstacles_objective_cost
```

将 `--objective cost` 改为 `--objective completion` 并使用另一输出目录即可对照。CLI 的缓存策略不变；首次运行可用新缓存目录，重新测量规划时添加 `--rebuild_cache`。优化目标写入 `summary.json`、`run_config.json`、报告、逐场景 CSV 和路线记录；验证器会检查目标一致性，旧结果未记录目标时按 `completion` 解释。

完成率更高并不保证 cost 更低：cost 同时受逾期任务数量和每个任务的相对逾期程度影响。算法间比较应保持数据、机器人数量、服务时间和补全规则一致。

## 10. 推理时间如何理解

统计的是 CPU 上**单个场景、完整搜索结束并形成完整路线的平均墙钟耗时**，包含距离/服务时间计算、搜索、剩余任务补全和路线提取。

不包含提前路径规划、缓存读写、数据读取与转换、Python 模块/动态库加载、C++ 编译、预热、独立路线重放校验、指标汇总和文件输出。障碍数据的推理阶段只把距离表转换为移动时间，不再建图或搜索路径。并行时，工作进程首次求解前会单独预热；进程启动和通信也不计入单次推理时间，但计入整批 `evaluation_wall_seconds`。额外的 `cpu_seconds` 是执行该求解的进程 CPU 时间，二者不要混用。

路径规划单独统计：`path_planning_seconds_total` 是规划进程累计计算时间；`path_planning_wall_seconds` 是规划阶段实际墙钟时间，包含缓存读写及进程开销。`evaluation_wall_seconds` 只统计后续评估循环，排除整个提前规划阶段。两种累计时间均不能直接当成并行阶段墙钟时间。

`--workers 1` 用于测量串行推理延迟；`--workers N` 用于加快整批评估。并行报告会记录实际进程数和资源竞争下的单次延迟，各次延迟之和不是整批完成所需时间。

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
python EILS/verify_results.py EILS/results/taptc96_python
```

程序重新读取路线，检查数据指纹、任务覆盖、机器人分配、完成时间、cost、完成率、种子、记录完整性和汇总，并写入 `verification.json`。障碍实验额外校验距离快照 SHA256、路线对应的缓存键、规划计时汇总；复算直接读取输出目录中的距离快照，不重新规划，删除共享缓存后仍可校验。原始数据仍须位于记录路径。

比较已有实验，参数顺序是 **C++ 目录在前，Python 目录在后**：

```bash
python EILS/compare_backends.py \
  EILS/results/taptc96 \
  EILS/results/taptc96_python
```

数据指纹、样本范围、算法列表、参数组、种子、服务时间、预算、重复次数和补全规则须一致。Python 的匹配模式可以与 C++ 的固定参数模式比较，只核对相同配置的对应记录。命令会更新 Python 目录下的对照 JSON 和 Markdown，不会重新搜索。

本机已保存的验证结果：192 个对应解无差异，完成时间最大绝对差为 0。时间来自两次独立实验，倍率只描述当时的实现效率。

## 13. 测试、文件职责与常见问题

```bash
python -m unittest discover -s EILS -t . -p 'test_*.py' -v
```

测试覆盖求解器、数据指纹、Python/C++ 一致性，以及串行/并行结果一致性和命令行进度输出。没有 `g++` 时，依赖编译器的跨后端核对测试会跳过，Python 求解仍可使用。完整记录见 [VALIDATION.md](VALIDATION.md)。

| 文件或目录 | 职责 |
|---|---|
| `evaluate.py` | 日常评测入口 |
| `solver.py` | 数据准备、后端选择、完整路线重放与指标校验 |
| `python_solver.py` | Python/NumPy 搜索实现 |
| `navigation.py` | 障碍数据读取、调用共享规划器与缓存、规划计时汇总 |
| `solver.cpp` | 保留的 C++ 搜索实现 |
| `data_identity.py` | 输入文件清单与 SHA256 校验 |
| `verify_results.py` | 复算已保存结果 |
| `compare_backends.py` | 比较两种后端的对应结果 |
| `test_parallel_evaluation.py` | 串行/多进程结果一致性、复算及命令行验证 |
| `test_navigation.py` | 避障距离、尺度与非活跃机器人、双后端、规划缓存复用和独立落盘复算 |
| `test_*.py`、`test_native.cpp` | 实现验证；日常评估无需直接运行 C++ 测试文件 |
| `SOURCES.md`、`VALIDATION.md` | 文献与适配说明、验证记录 |
| `results/` | 正式实验与调试归档，先看其中的索引 |
| `build/`、`__pycache__/` | 构建、诊断与 Python 缓存，不是实验结果 |
| `*.log` | 运行日志；最终指标以对应结果目录为准 |

| 遇到的情况 | 处理方法 |
|---|---|
| `python` 找不到或缺少 NumPy/PyTorch | 按第 2 节执行 `conda activate mrta_c`，并检查该环境中的项目依赖 |
| 启动后长时间没有新进度 | 使用 `--progress_every 1`；每个场景的全部配置完成后才更新进度，Python 正式搜索可能较慢 |
| 想加快整批评估 | 使用 `--workers 8` 并按可用 CPU/内存调整；切换运行方式须使用新输出目录 |
| `Output directory is not empty` | 换一个新目录名；不要覆盖正式实验 |
| 输出目录被拒绝 | 将 `--output_dir` 放在 `EILS` 内 |
| 随机 1000 场景使用匹配模式时报错 | 去掉 `--match_taptc_groups`，显式指定参数组 |
| 完整 TAPTC 匹配模式仅指定 tight 后报错 | 保留 `--profiles tight slack`，使 group2 也有参数 |
| 只有部分 CSV/JSONL 和 `run_config.json` | 可能仍在运行或已中断；未完成的记录不能当最终结果 |
| `models.ils_tight.scenarios` 只有 48 | 匹配模式下这是 group1 统计；96 场景总体看 `matched_models.ils` |
| Python/C++ 耗时差几十倍 | 确认后端与预算，不混用两种实现的耗时 |
| 完成率和 cost 同时升高 | 两个指标含义不同，见第 9 节 |
| 想与原论文时间直接对应 | 当前尚未严格复现其运行效率与计时细节，见 `SOURCES.md` |
