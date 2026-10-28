# ILS / EILS 使用入口

本目录提供 ILS、EILS 的固定数据集评估。**默认使用 Python/NumPy 后端**；可用 `--backend cpp` 切换至保留的 C++ 实现。

支持 `z_test_path/data/n100_obstacles_example.pkl`：自动使用可见图 + Dijkstra 提前规划，后续搜索与指标复算均查缓存距离表。结果分别输出 cost、任务完成率、推理时间和路径规划时间。完整命令和缓存说明见[障碍数据用法](README.zh-CN.md#障碍数据可见图--dijkstra)。

| 需要做什么 | 从哪里开始 |
|---|---|
| 环境、运行命令、参数与指标说明 | [中文使用说明书](README.zh-CN.md) |
| 查找正式实验，区分 Python/C++ 与调试结果 | [结果目录索引](results/README.md) |
| 查阅文献来源、算法规则与适配边界 | [SOURCES.md](SOURCES.md) |
| 查阅测试覆盖与后端一致性验证 | [VALIDATION.md](VALIDATION.md) |

## 先看已有结果

- **原始 TAPTC 96 场景，Python**：[评测报告](results/taptc96_python/REPORT.md)、[summary.json](results/taptc96_python/summary.json)。这是目前推荐查看的 Python 结果。
- **相同 TAPTC 场景，历史 C++**：[评测报告](results/taptc96/REPORT.md)。
- **训练分布匹配的 1000 场景，历史 C++**：[评测报告](results/n100_train_matched_1000/REPORT.md)。目前没有对应的 Python 全量实验结果。
- **Python/C++ 结果一致性**：[后端对照报告](results/taptc96_python/BACKEND_COMPARISON.md)。192 个对应解的路线、完成时间、完成数、cost 和迭代统计全部一致。

## 运行一次新的 TAPTC 96 场景实验

在同一终端执行；本机已安装依赖的 Python 位于 `mrta_c` 环境。`run01` 是新实验目录示例，若已存在结果，请换成未使用的名称。

```bash
cd /home/hustcsib/chiu_code/CapAM-MRTA
conda activate mrta_c

python EILS/evaluate.py \
  --backend python \
  --data data/mrta_gini \
  --match_taptc_groups \
  --algorithms ils eils \
  --profiles tight slack \
  --seed 20261025 \
  --output_dir EILS/results/taptc96_python_run01

python EILS/verify_results.py EILS/results/taptc96_python_run01
```

该命令在 group1 使用 tight 参数、group2 使用 slack 参数。每个场景各运行 ILS 和 EILS 一次，共 **192 次求解**。输出均在指定目录中，程序拒绝覆盖非空目录。完整选项、1000 场景命令和小样本验证见 [使用说明书](README.zh-CN.md)。

需要加快整批评估时，增加 `--workers 8 --progress_every 1`，并使用一个新的输出目录。默认 `--workers 1` 为串行；并行时保持相同种子和搜索预算，各次求解延迟之和与整批墙钟时间分别记录。完整命令见 [多进程并行说明](README.zh-CN.md#多进程并行加速)。

## 阅读结果前需要明确

- 默认 `--objective completion` 优化**按时完成任务数**；`--objective cost` 最小化每轮候选方案补全全部任务后的 cost。两个后端均支持，同值保留先找到的方案。
- cost 模式保留原有可行插入和扰动邻域，在每轮补全评分后恢复可行搜索状态；它是目标改版，不是论文原始目标。ILS 按对应目标的连续无改进次数停止，EILS 按构造次数停止。
- `tight/slack` 是搜索参数组；原始 TAPTC 另有真实截止时间分组。1000 场景数据没有这种分组，详见说明书。
- 当前耗时是**完整搜索结束、生成完整路线的时间**，包括剩余任务补全；它不是“首次找到最好解”的时间，也不是机器人的物理执行时间。
- 本实现依据公开文献重写并适配当前环境。Python/C++ 一致性不等于作者原版一致性，当前尚未严格复现论文的运行时间与速度排序。
