# BiG-MRTA 目标对比

数据：`/home/hustcsib/chiu_code/CapAM-MRTA/z_test_path/data/n100_obstacles_example.pkl`，100 场景；`capam_full_visit`，CPU 1 线程，预热 1 次。

路径：`visibility_dijkstra`，epsilon=1e-06；两个目标逐场景使用相同距离缓存，均访问所有任务。

| 配置 | 平均 cost ↓ | 按时完成率 ↑ | 推理 ms/场景 | 规划总秒数 | 缓存命中 |
|---|---:|---:|---:|---:|---:|
| completion | 111.237318 | 72.390% | 192.120 | 0.000000 | 100/100 |
| cost | 343.591554 | 39.800% | 63.646 | 0.000000 | 100/100 |

`cost` 相对 `completion`：平均 cost 差值 +232.354236；按时完成率差值 -32.590 个百分点。

逐场景 cost：0 个降低，0 个持平，100 个升高（绝对容差 1e-6）。

`completion` 保留原版 deadline 可行边和激励权重，是完成率启发式。`cost` 对所有剩余任务建边，用 Hungarian 最小化本轮预计新增惩罚：按时为 0，逾期为 finish/deadline；平局按 SciPy 固定行/任务顺序处理。它是局部匹配，未计入本轮决策对后续任务的影响，不保证最终总 cost 更低，也不保证完成率更高。两种配置均使用相同 cost 和实际 deadline 统计。

推理时间排除规划、缓存 I/O、预热、额外重放及结果写入。两个目标依次运行，避免并发测时干扰；运行顺序为 completion、cost，单次计时可能波动。缓存读写耗时见各目标 summary.json。

各目标 `audit.json` 保存全部场景路线重放及指标核验；`paired_scenarios.csv` 保存逐场景差值。
