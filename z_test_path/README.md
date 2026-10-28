# 带障碍物的 MRTA 测试集

`generate_test_set.py` 使用用户指定的 ASCII 地图，随机生成任务点、机器人起点和一个仓库。所有坐标都严格位于可通行栅格内部。

## 指定地图

编辑 [maps/example.txt](maps/example.txt)，或新建文本文件：

```text
..........
..###.....
..###.....
..........
.....###..
.....#....
.....#....
..........
.###......
..........
```

- `#` 表示整个栅格都是障碍物，`.` 表示可通行栅格。
- 每行长度必须相同，只允许这两个字符；不能有空行、空格或注释，必须至少有一个 `.`。允许文件末尾换行。
- 字符矩阵铺满整个正方形地图；行数、列数不同则每格是长方形。希望每格也是正方形时，使用相同行数和列数。
- 坐标原点在左下角，x 向右、y 向上；文本第一行对应地图最上方。
- 对于 `H` 行、`W` 列、物理边长 `L` 的地图，第 `r` 行、第 `c` 列对应 `x∈[cL/W,(c+1)L/W]`、`y∈[(H-r-1)L/H,(H-r)L/H]`，索引从 0 开始。采样点严格处于栅格内部，不在边界上。

### 内置典型地图

`maps/` 另外提供 12 张统一为 **20×20** 的地图：空场、中心方块、稀疏/密集障碍、柱阵、仓库货架、窄/宽瓶颈、四房间、U 形障碍、蛇形通道和九房间办公室。全部新增地图的自由区域连通，可用于当前的两两路径预计算。

查看[地图索引与生成命令](maps/README.md)或[总览图](maps/overview.png)。选择新地图后需重新生成 `.pkl`，已有数据集保存的是原地图。

另有 **12 张 20×20 随机地图**：随机散点和随机矩形两种风格，障碍占比为 10% / 20% / 30%，每组两个种子，均保持自由区域连通。查看[随机地图总览](maps/random_overview.png)和[地图索引](maps/README.md#随机地图)。可用 `generate_random_maps.py --seed 20261101` 在默认地图目录中继续生成一组新地图。

## 生成数据

在项目根目录、已安装项目依赖的环境中运行：

```bash
conda activate mrta_c
python z_test_path/generate_test_set.py \
  --map_file z_test_path/maps/example.txt \
  --graph_size 100 \
  --robot_counts 2 3 4 5 6 7 \
  --dataset_size 1000 \
  --seed 20261026 \
  --filename z_test_path/data/n100_obstacles.pkl
```

每个场景独立随机生成任务点、机器人起点和仓库；先等概率选择可通行栅格，再在格子内部连续均匀采样，等价于在整片可通行区域上均匀采样。坐标不局限于格子中心。所有机器人槽位（包括该场景未启用的槽位）都避开障碍物。

与原生成器相比，**仓库也会每个场景重新随机生成，不再固定为 `(0,0)`**。本生成器只限制点不落入障碍物，不要求它们位于同一连通区域，布局是否连通由指定地图决定。

参数沿用 `z_test/generate_data.py`：`--graph_sizes 50 100 200` 可生成多个任务规模；`--n_agents 5` 可固定机器人数量；支持地图边长、速度、截止时间、工作量、能力和缩放参数。`--map_file` 也可写为 `--obstacle_map`。已有 `.pkl` 或 `.json` 时默认拒绝覆盖，使用 `--overwrite` 才会替换。

默认 `map_size=100`、`coordinate_scale=100`，物理坐标在 `[0,100)`，保存后的模型坐标在单位正方形 `[0,1)`。字符栅格边界也按同样比例换算。若使用 `--map_size 1 --coordinate_scale 1`，速度等物理参数也应按所需单位设置。

## 绘制场景

`plot_test_set.py` 读取数据文件中保存的地图，画出障碍物、任务点和实际启用的机器人起点，并保存图片：

```bash
python z_test_path/plot_test_set.py \
  --data z_test_path/data/n100_obstacles_example.pkl \
  --index 0
```

默认保存到 `z_test_path/figures/<数据集名称>_scene0.png`。`--index` 从 0 开始，表示数据集中的第几个场景。蓝色圆点代表任务，橙色三角形代表机器人，深色方块代表障碍物。机器人标签 `R1`、`R2` 等从 1 开始，不显示未启用的机器人槽位。

默认使用归一化模型坐标；使用 `--units physical` 显示物理坐标。添加 `--show_ids` 显示从 1 开始的任务编号，`--no_grid` 隐藏栅格线。默认绘制初始位置，提供 `--routes` 可叠加评估产生的实际绕行路线。

```bash
python z_test_path/plot_test_set.py \
  --data z_test_path/data/n100_obstacles_example.pkl \
  --index 1 --units physical --show_ids \
  --output z_test_path/figures/scene1_physical.png
```

`--output` 支持 `.png`、`.pdf`、`.svg`，已有图片需要添加 `--overwrite` 才能覆盖。绘图使用 Matplotlib，可在没有图形桌面的服务器上运行。

### 绘制指定场景的规划路线

`plot/plot_plan.py` 根据测评输出的 `routes.json` 绘制机器人执行路线。默认读取示例数据集和 `results/capam_path_100/routes.json`，只需指定场景 index：

```bash
python z_test_path/plot/plot_plan.py --index 0
```

图片包含障碍物、任务点、机器人起点和避障路径；不同机器人使用不同颜色，箭头表示行进方向，图例给出每个机器人分配的任务数量，不绘制仓库。`--index` 从 0 开始，选择数据集中的场景，不是任务编号。

默认输出到 `z_test_path/plot/figures/<测评结果目录名>/<数据集名称>_scene<index>_plan.png`。例如 index 0 的 CapAM 图片为 `plot/figures/capam_path_100/n100_obstacles_example_scene0_plan.png`。

选择其他数据集、模型测评结果或输出路径：

```bash
python z_test_path/plot/plot_plan.py \
  --data z_test_path/data/n100_obstacles_example.pkl \
  --routes z_test_path/results/am_path_100_epoch99/routes.json \
  --index 3 --show_ids \
  --output z_test_path/plot/figures/am_scene3.png
```

支持 `--units physical`、`--no_grid`、`--dpi` 和 `--overwrite`，图片格式支持 PNG/PDF/SVG。路线文件必须来自同一份数据集并包含所选 index；程序校验数据集指纹。绘图直接读取已保存的规划路线，不重新运行模型或路径搜索。

## 输出与评估

- `.pkl`：沿用 `capam_mrta` 版本化场景格式，在文件顶层的 `obstacle_map` 中保存完整地图及坐标约定。
- 同名 `.json`：保存地图、参数、种子、采样分布和单位，便于直接查看。
- 地图内容已写入输出，后续编辑原地图文件不会改变已生成的数据集。
- 相同参数、地图和种子在相同运行环境下生成相同的场景。

### 使用已经训练好的 GCAPCN、AM 或 Mamba 测评障碍物场景

`z_test_path/evaluate.py` 支持两种路径规划算法，由 `--path_planner` 选择。默认使用原有的栅格 A*；也可使用可见图 + Dijkstra。评估先用选定算法预计算路径，再在解码、完成时间计算、候选方案选择和路线重放中统一查表。GCAPCN、AM 和 Mamba 均支持这两种算法，无需重新训练。

```bash
python z_test_path/evaluate.py \
  --model outputs/mrta_100/capam_paper_20261023T154551842127 \
  --data z_test_path/data/n100_obstacles_example.pkl \
  --device cpu --eval_batch_size 10 \
  --path_planner grid_astar \
  --path_workers 8 \
  --output_dir z_test_path/results/capam_path_100
```

模型目录默认取编号最大的 `epoch-*.pt`（上述目录当前为 `epoch-99.pt`），也可直接传具体检查点文件。默认测评数据集所有场景；添加 `--num_samples 3` 可先检查前三个，`--offset 10 --num_samples 5` 则检查 index 10～14。结果目录已有文件时需要 `--overwrite`。

入口会从检查点的 `args.json` 自动识别编码器，无需额外指定 `--encoder`。AM 和 Mamba 使用相同数据、规划缓存、查表仿真器及 cost 定义，例如：

```bash
python z_test_path/evaluate.py \
  --model outputs/mrta_100/am_rl_100_20261024T154131715750 \
  --data z_test_path/data/n100_obstacles_example.pkl \
  --device cpu --eval_batch_size 10 \
  --output_dir z_test_path/results/am_path_100

python z_test_path/evaluate.py \
  --model outputs/mrta_100/mamba_rl_100_20261024T160256102667 \
  --data z_test_path/data/n100_obstacles_example.pkl \
  --device cpu --eval_batch_size 10 \
  --output_dir z_test_path/results/mamba_path_100
```

目录形式会随新增训练检查点选择最新 epoch；需要重复实验时请指定完整的 `epoch-XX.pt` 路径。不同模型可以复用同一场景的路径缓存。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--path_planner` | `grid_astar` | `grid_astar`：栅格 A*；`visibility_dijkstra`：可见图 + Dijkstra |
| `--path_subdivisions` | `10` | 仅用于 `grid_astar`：每个字符格的每条边细分数量，10×10 字符地图对应 100×100 规划网格 |
| `--visibility_epsilon` | `1e-6` | 仅用于 `visibility_dijkstra`：障碍物角点向自由空间偏移的距离，以内部 `[0,1]²` 单位地图计 |
| `--path_workers` | `1` | CPU 路径预计算进程数，可按可用 CPU 核数增加 |
| `--cache_dir` | `z_test_path/cache` | 每个场景的距离及路径缓存目录 |
| `--rebuild_cache` | 关闭 | 重新计算并替换相应缓存 |
| `--device` | `auto` | 模型推理设备：`cpu`、`cuda`、`cuda:N` 或自动选择 |
| `--cpu_threads` | `1` | CPU 模型推理线程数 |
| `--eval_batch_size` | `1` | 同时测评场景数；混合任务规模时使用 1 |
| `--decode_strategy` | `greedy` | `greedy` 或 `sample` |
| `--width` | `1` | `sample` 模式下每个场景的候选方案数 |
| `--warmup_batches` | `1` | 不计入推理耗时的预热次数 |

### 路径规划算法与切换示例

| 选项 | 实现 | 搜索节点 | 返回路径 |
|---|---|---|---|
| `grid_astar` | `cankao_planner.GridPathPlanner` | 细分后可通行栅格的中心 | 八邻接 A* 搜索后进行碰撞检测和直线平滑 |
| `visibility_dijkstra` | `visibility_planner.VisibilityGraphPlanner` | 向外偏移的障碍物角点，以及真实起终点 | 可见边按欧氏长度加权，Dijkstra 返回图上的最短折线 |

两种算法读取同一份 ASCII 障碍地图；可见图方法直接使用障碍矩形的几何信息，不生成细分搜索网格。`#` 对应闭合障碍矩形，接触边界或角点也判为碰撞；机器人按点处理，不能穿过两个障碍物接触的角点。直线无遮挡时，两种算法都直接返回欧几里得线段。起终点保留原始连续坐标，不吸附到栅格中心。

**栅格 A* 方法**沿用参考方法：八邻接搜索、禁止对角穿角、真实起终点连接、逐段碰撞检查的直线平滑。绕行长度受 `--path_subdivisions` 影响，不宣称是连续空间中的精确最短距离。原有命令不增加参数时继续使用此方法。

**可见图 + Dijkstra** 的步骤：

1. 将每个障碍矩形的四个角点分别沿 x、y 方向向外偏移 `visibility_epsilon`；删除地图外、落入其他障碍物或接触其边界的候选点，并去重。
2. 对候选点两两做完整线段碰撞检测，无遮挡时连接无向边，边权为欧氏距离。相邻障碍格的内部角点因此不会成为可用节点。
3. 将真实起点和终点分别连接到可见的候选点。使用 Dijkstra 搜索，保留真实起终点和中间角点，距离取最终折线的弧长。
4. 每个场景内只建立一次障碍物可见图，并复用同一坐标的可见连边和同一起点的搜索树。模型推理阶段仍然只查预计算表。

由于现有碰撞规则不允许贴着障碍边界行走，角点必须有微小偏移。Dijkstra 保证的是**上述有限可见图上的最短路径**，不宣称是零偏移连续空间中的精确最短路径。`visibility_epsilon` 必须有限，至少为 `4 × float64 机器精度`（约 `8.88e-16`），且小于最窄字符格边长的一半，即 `< 0.5 / max(地图行数, 地图列数)`。默认 `1e-6` 通常无需调整；偏移过大可能使狭窄通道无法用图连接，此时可减小该值。

该参数是**候选角点的偏移量**，不是机器人半径，也不保证整条路径具有相同的障碍净空。内部单位地图上每个坐标轴偏移 `epsilon`，对应物理坐标的 `epsilon × map_size`、模型坐标的 `epsilon × map_size / coordinate_scale`。规划器输出的距离和路径会换回数据集的模型坐标，速度也使用保存的模型单位，时间仍为秒。

下面固定同一个检查点、数据集和解码参数，用两个结果目录比较规划算法：

```bash
# 栅格 A*：与现有默认算法相同
python z_test_path/evaluate.py \
  --model outputs/mrta_100/capam_paper_20261023T154551842127/epoch-99.pt \
  --data z_test_path/data/n100_obstacles_example.pkl \
  --path_planner grid_astar --path_subdivisions 10 \
  --device cpu --eval_batch_size 10 --path_workers 8 \
  --decode_strategy greedy --seed 20261026 \
  --output_dir z_test_path/results/capam_grid_astar

# 可见图 + Dijkstra
python z_test_path/evaluate.py \
  --model outputs/mrta_100/capam_paper_20261023T154551842127/epoch-99.pt \
  --data z_test_path/data/n100_obstacles_example.pkl \
  --path_planner visibility_dijkstra --visibility_epsilon 1e-6 \
  --device cpu --eval_batch_size 10 --path_workers 8 \
  --decode_strategy greedy --seed 20261026 \
  --output_dir z_test_path/results/capam_visibility_dijkstra
```

可以给两条命令都增加 `--num_samples 3` 先做小规模验证。AM、Mamba 只需替换 `--model` 和 `--output_dir`。`--path_subdivisions` 不影响可见图，`--visibility_epsilon` 不影响 A*；输出中不适用的参数在 JSON 中为 `null`、在 CSV 中为空。切换算法会改变距离及移动时间，GCAPCN 的建图空间距离也随之改变，因此模型可能产生不同的任务顺序和完成率。

可见图的建图成本随候选角点数量的平方增长，每条候选边还需要碰撞检查；A* 的成本受细分网格规模影响。两者快慢依赖障碍形状、数量和查询规模，应根据实际测评比较。`--path_workers` 对两种算法都有效，路径规划均在 CPU 上运行。

绘制可见图算法评估后的路线：

```bash
python z_test_path/plot/plot_plan.py \
  --data z_test_path/data/n100_obstacles_example.pkl \
  --routes z_test_path/results/capam_visibility_dijkstra/routes.json \
  --index 0
```

### 预计算、缓存与模型行为

预计算包含 `task_to_task[N,N]` 和 `robot_to_task[机器人槽位数,N]`，只规划任务及实际启用的机器人起点，未启用槽位被填充并屏蔽。仓库仅保留原网络需要的节点字段，不作为实际行程的出发点或终点。存在不可达点对时报告场景 index 并停止，不跳过场景，也不回退为直线距离；生成器不会为此自动移动已有点。

缓存保存在单独文件中，原始数据集仍记录 `path_distances_precomputed=false`。缓存依据地图、坐标、有效机器人数量、归一化地图边长、规划算法、算法有效参数及规划器源码指纹区分：A* 使用细分参数，可见图使用角点偏移参数。不适用于当前算法的参数不影响缓存键。同一 `--cache_dir` 可同时保存两种算法的缓存，也可在不同模型之间复用同一种算法的缓存，切换算法不会误用另一算法的距离。当前缓存格式版本为 2，旧版缓存会自动忽略并重新计算，无需手动删除。首次运行可能主要耗时在路径预计算，后续相同场景可直接复用；命中可见图缓存时也不会重新建图。

GCAPCN 原建图距离包含 `(x,y,归一化截止时间)` 三个分量。适配器保留输入特征和全部已训练参数，把空间分量换成路径距离，即 `sqrt(path_distance² + normalized_deadline_difference²)`。AM 和 Mamba 编码器没有显式的几何距离建图，保留其坐标、归一化截止时间输入和全部已训练参数；注意力中的特征相似度不是欧几里得距离。本次适配没有给 AM/Mamba 编码器增加障碍图或路径距离特征。

三种编码器共用 `AttentionModel` 解码器，并全部接入 `PathMRTA`：仿真中移动时间直接使用 `path_distance / max_speed`，工作时间和原 cost 定义沿用现有规则。这是已有权重在障碍物环境中的直接迁移测评，调度质量由实际测评结果决定。`summary.json` 的 `encoder_uses_path_distances` 表示编码器本身是否读取路径距离，AM/Mamba 为 `false`，三者的移动计时均使用缓存路径距离。

输出文件：

| 文件 | 内容 |
|---|---|
| `REPORT.md` | 中文汇总：规划算法及有效参数、总体和按机器人数量分组的指标、CPU/GPU 设备，以及三项时间 |
| `overall.csv` | 一行总体指标、设备、推理总时间、平均每样本推理时间和路径规划时间 |
| `summary.json` | 规划算法及参数、完整总体统计、`by_n_agents` 机器人数量分组、时间字段定义、独立重放误差 |
| `groups.csv` | 每个 `n_agents` 一行，含样本数、平均完成率/cost、设备及同样三项时间 |
| `scenarios.csv` | 每个原始 index 的指标、实际设备、所属 batch、路径规划时间及推理分摊时间 |
| `batches.csv` | 每个真实推理 batch 一行，记录该批样本数和推理时间 |
| `routes.json` | 规划算法及参数、每段实际绕行路径、移动及完成时间，机器人和任务 ID 从 1 开始 |

一个“样本”就是数据集中的一个场景。`completion_percent` 是按任务数量加权的按时完成率；`mean_scenario_completion_percent` 是各场景完成率的算术平均。`mean_cost` 是场景 cost 的算术平均。固定每场景 100 个任务时，两种平均完成率相等。总体、每个机器人数量分组都同时输出两种完成率。

设备与主要推理字段：

| 字段 | 含义 |
|---|---|
| `device_type` / `device` / `device_name` | 实际使用的 CPU/GPU 类型、设备编号和硬件名称 |
| `requested_device` | 命令行请求的 `auto`、`cpu`、`cuda:N` 等 |
| `path_device` | 路径规划设备，目前为 CPU |
| `path_planner` / `planner` | 算法配置值及实现说明，同时写入各 CSV、`summary.json`、`routes.json` 和报告 |
| `path_subdivisions` / `visibility_epsilon` | 当前算法的有效参数；不适用的参数为 `null`（JSON）或空值（CSV） |
| `inference_seconds_total` | 正式推理总耗时，包含编码、解码、查表仿真和候选选择 |
| `inference_seconds_per_scenario` | 平均每样本推理耗时，秒 |
| `path_planning_seconds_total` | 各样本实际建图、路径搜索、平滑（适用时）和距离表计算的累计耗时，秒 |
| `inference_timing_mode` | `single_scenario_latency` 为逐场景延迟，`batch_amortized` 为 batch 分摊时间 |

总体和分组只保留上述三项时间，单位均为秒。路径规划时间为各样本/进程实际计算的累计耗时；并行时不是整个规划阶段的实际等待时间。命中缓存时不重新规划，规划时间为 0，不计缓存读写；需要重新测量规划时间时使用 `--rebuild_cache`。

**batch size 大于 1 时，平均每样本和各机器人数量分组的推理时间，按每批实际样本数均摊。** 需要逐场景延迟时使用 `--eval_batch_size 1`。整批真实推理时间在 `batches.csv` 只记录一次。

GPU 推理计时前后同步 CUDA。推理时间不包含规划、缓存读取、预热、设备传输、重放和文件导出。

绘制评估后的路线，颜色对应各机器人：

```bash
python z_test_path/plot_test_set.py \
  --data z_test_path/data/n100_obstacles_example.pkl \
  --routes z_test_path/results/capam_path_100/routes.json \
  --index 0
```

默认保存为 `z_test_path/figures/<数据集名称>_scene0_routes.png`。路线文件校验数据集内容指纹，防止把旧路线画到修改后的数据上。

**原 `z_test/evaluate.py` 仍只计算欧几里得移动距离；障碍物测评应使用本目录的 `evaluate.py`。**

## 验证

```bash
python -m unittest z_test_path.test_generate_test_set
python -m unittest z_test_path.test_path_evaluate
python -m unittest z_test_path.test_visibility_planner
python -m unittest z_test_path.test_reporting
```

测试覆盖单障碍解析路径、全部 2×3 障碍布局的连通性、边界碰撞和对角穿角、无障碍和零长度路径、坐标缩放、算法切换与缓存隔离，以及三种编码器的 greedy/sample 评估、并行路径预计算和输出中的算法记录。
