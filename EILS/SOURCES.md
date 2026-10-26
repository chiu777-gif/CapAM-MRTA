# 原始文献、算法对应与复现边界

本文用于核对来源和适配细节。运行方法见 [中文使用说明书](README.zh-CN.md)，已有实验身份见 [结果目录索引](results/README.md)。

## 来源

1. Paul, Ghassemi, Chowdhury, *Learning Scalable Policies over Graphs for Multi-Robot Task Allocation using Capsule Attention Networks*, ICRA 2022，§II-A、§IV-A2，公式 (1)。
   - https://arxiv.org/abs/2205.03321
   - https://arxiv.org/html/2205.03321
   - DOI: https://doi.org/10.1109/ICRA46639.2022.9812370
   - 明确将 ILS 对应至参考文献 [20]，EILS 对应至 [19]；没有给出这两种基线的完整超参数和适配实现。
2. Mitiche, Boughaci, Gini, *Iterated Local Search for Time-extended Multi-robot Task Allocation with Spatio-temporal and Capacity Constraints*, Journal of Intelligent Systems 28(2):347–360, 2019。
   - DOI / 出版方：https://doi.org/10.1515/jisys-2018-0267
   - 作者公开的原文全文：https://www.researchgate.net/publication/329843114_Iterated_Local_Search_for_Time-extended_Multi-robot_Task_Allocation_with_Spatio-temporal_and_Capacity_Constraints
   - 作者出版物目录：https://www-users.cse.umn.edu/~gini/publications/class_rescat.html
   - 采用 §4.1–4.3 的公式 (6)–(12)、算法 1–3，参数取 §5.2。
3. Vansteenwegen, Souffriau, Vanden Berghe, Van Oudheusden, *Iterated local search for the team orienteering problem with time windows*, Computers & Operations Research 36(12):3281–3290, 2009。
   - https://doi.org/10.1016/j.cor.2009.03.008
   - https://biblio.ugent.be/publication/1143660
   - 原文出版页可访问；仓储全文标注机构内访问，未将其冒充为已取得的原文全文。
4. 相同作者的 *Metaheuristics for Tourist Trip Planning*，§4.1–4.3、算法 3：作者对相同 ILS 的详细说明和伪代码，作为 ILS 操作步骤的交叉核对来源。
   - https://www.researchgate.net/publication/226088125_Metaheuristics_for_Tourist_Trip_Planning
   - 核对 Insert、Shake、无改进次数停止条件、当前解继续搜索、改善后重置删除量、循环更新删除起点。

查阅日期：2026-10-25。也检查了 CapAM 作者公开仓库 `adamslab-ub/CapAM-MRTA` 的文件树，提交 `7a3800c3e0657b03b20bda5070d2d8a8f9c44f66` 中没有 ILS/EILS 基线实现。因此这里是**依据公开文献重新实现并适配当前环境**，不能声称逐行复现作者未公开的基线代码，或精确复现论文中的基线数值。

## Python 与 C++ 后端

默认后端已改为 `python`：`python_solver.py` 用 Python 实现搜索控制、路线构造、扰动、接受规则和 MT19937-64，用 NumPy 同时计算一个机器人的插入候选，不调用 C++ 求解器，不需要 `g++`、Numba 或 Cython。`--backend cpp` 保留原有 `solver.cpp` 实现用于核对与快速运行。

两种后端保留相同的任务/机器人/位置并列顺序、浮点加减顺序、原始服务时间选项、搜索预算和剩余任务补全规则。Python MT19937-64 使用 C++ 标准引擎的参考序列验证，包括 312 个状态字更新前后的边界，确保给定种子产生同样的删除起点。后端切换不会调整迭代上限以迁就原论文的时间或速度排名。

历史结果目录 `n100_train_matched_1000` 和 `taptc96` 均为 C++ 后端；新结果单独保存，并记录后端和对应源代码指纹。`compare_backends.py` 对相同场景、种子和参数逐项核对路线、完成时间、完成数、cost、实际迭代数与最好解所在迭代数；计时倍率描述具体实现效率。

`--match_taptc_groups` 在真实 group1/group2 数据上为每个场景仅运行对应参数一次；旧实验在每个场景上运行两套参数再按输入分组汇总。两种运行方式使用相同的场景全局索引和派生种子，后端核对只选对应的实际记录。

## 构造步骤：两种算法共用

每个机器人维护独立路线。对所有未分配任务，遍历每条路线的所有插入位置。任务必须在自身截止时间前完成；插入延迟不能超过后续所有任务的最小剩余时间。选择延迟最小的插入；持续至全部任务可行或不存在可行插入。

在任务分数均为 1、没有开放时间和强制终点的 TAPTC 中，ILS 的 `score² / shift` 和 EILS 的 `1 / shift` 是同一排序。初始位置是数据中各机器人的真实起点；没有返回仓库要求。

对应实现：`Search::times` 为公式 (7)，`Search::candidates` 为公式 (6)、(8)–(11)，`Search::construct` 为算法 1。路线更新后重算其时间和最大允许延迟；这与逐项增量更新具有相同数学含义，便于审查。候选缓存只复用未改变的路线；每轮扰动后全部重新生成。

## 扰动与接受规则

- **ILS**：从每条路线相同的循环起点删除连续 `r` 个任务；每轮将起点增加 `r`，并将 `r` 增加 1。起点按扰动后最短路线长度回绕；达到删除量阈值后重置为 1。全局最好完成数严格增加时，将无改进次数和 `r` 重置。停止条件为连续 `maxTrials` 轮无严格改进，或已完成全部任务。下一轮使用当前解，不回退到最好解。
- **EILS**：每轮从各条路线独立随机选择起点，循环删除 `floor(score(S) × coefficient / robots)` 个任务；使用当前解继续搜索。严格改善完成任务数才更新最好解。停止条件为最多 `maxIter` 次构造，或已完成全部任务。
- 两者的优化目标均保留原始的**按时完成任务数**。没有改成 CapAM cost，也没有用 cost 在完成数相同的解之间择优。

## 参数

| 参数组 | ILS maxTrials | ILS 删除量阈值系数 | EILS maxIter | EILS 删除比例 |
|---|---:|---:|---:|---:|
| tight / 原文 group 1 | 600 | 1/2 | 3000 | 1/3 |
| slack / 原文 group 2 | 600 | 2/5 | 5000 | 1/8 |

取自 Mitiche et al. §5.2。这里的固定数据集采用连续均匀截止时间，不属于该文献的 tight/slack 分组。因此在相同的全部 1000 个场景上分别运行两套参数；**tight/slack 是参数名称，不是给该数据集重新分组，也没有根据测试结果选参**。

对原始 `data/mrta_gini` 96 个场景，输入确有 `group1`、`group2` 两个截止时间分组。`matched_models` 根据输入分组选择对应参数：group1 取 tight，group2 取 slack，不根据完成率或 cost 选参。历史 C++ 实验在全部场景上运行两套固定参数，再从记录中提取每种算法的 96 场景匹配汇总；正式 Python 实验使用 `--match_taptc_groups`，直接在每个场景运行对应参数，每种算法同样覆盖全部 96 个场景。

ILS 阈值采用 `max(1, floor(N × coefficient / robots))`，每轮增加删除量后达到阈值即重置；这沿用作者伪代码的阈值判断。删除任务数不超过当前路线长度。

## 必须明确的适配及未公开细节

1. **连续时间**：Mitiche et al. 的服务时间为 `ceil(workload / capacity)`；CapAM 和本仓库为 `workload / capacity`。默认选 `continuous` 与当前 CapAM 保持一致，现有正式结果均使用该模式。`--service_time ceil` 对加载后的数值比值向上取整；归一化浮点数在整数边界的舍入会影响结果，因此该选项不直接等于按原始物理文本取整的作者原版实现。两种服务时间模式的结果不能混用。
2. **完整覆盖 cost**：原始 ILS/EILS 可跳过不可行任务，原生解中未访问任务没有完成时间，不能直接套用 CapAM 的 cost。这里先冻结其最优可行路线，然后对剩余任务重复：选择最早空闲的机器人，追加旅行时间与服务时间之和最小的任务。所有任务恰好访问一次后，报告 `sum(finish/deadline if finish > deadline else 0)`。这是本实现明确给出的统一适配，**CapAM 论文没有公开其基线使用的补全规则**。
3. **完成率不依赖补全**：最优可行解已经无法插入任何剩余任务，因此补全不应增加按时完成数。每次求解均独立重放完整路线，并验证最终按时完成数恰好等于原生搜索得分。补全选择会影响 cost，不能把它作为论文精确复现的 cost。
4. **调度语义**：原始方法优化各机器人的显式路线，允许机器人在完成自己的路线后空闲；不把路线展平成一串任务再强行交给公共模拟器的“最早空闲机器人”。后者会改变机器人分配。独立路线重放使用同一欧氏旅行时间和服务时间公式，单机器人情形另与公共模拟器核对。
5. **整数化与边界**：文献没有规定 EILS 非整数删除量的舍入方式；这里向下取整并保证至少删除一个任务，以避免 `r=0` 停滞。空路线跳过，删除量截断至路线长度；起点回绕用取模实现。
6. **并列与随机数**：相同插入延迟按任务 ID、机器人 ID、位置顺序择首；相同完成数保留最先发现的解。两种后端均使用 MT19937-64 序列，C++ 调用标准引擎，Python 实现相同序列；随机起点使用输出对路线长度取模。种子和每个场景的派生规则写入输出。原作者 Java 随机数实现与上述边界规则未公开，不能保证与作者实现逐比特一致。
7. **重复次数**：默认每个配置每场景一次固定种子运行，得到 1000 个场景的平均值；原始 EILS 论文还做过 10 次重复。可用 `--repeats 10`，输出保留每次重复，不进行最优一次挑选。
8. **推理时间**：报告当前硬件上完整搜索的实测总耗时；没有用“首次找到最好解”的时刻代替实际算法停止时刻。单线程、逐场景运行，包含补全与路线提取，排除编译、读数据、转换、预热、结果校验和文件输出。论文硬件、语言、参数和计时口径并不相同，不以论文表格耗时校准实现。
9. **数值精度**：保留数据文件中已保存的坐标、速度、工作量和能力数值，不重新抽样或归一化。搜索和独立重放使用 float64，截止条件为精确的 `finish <= deadline`，不加容差；完成时间一致性验证才使用数值容差。CapAM 公共模拟器使用 float32，临界边界的舍入行为可能不同。

如需与 CapAM/AM 的推理时间比较，应重新用同一 CPU、相同线程数和 batch size=1 运行模型。已有 GPU batch size=32 的结果是批量均摊耗时，不能直接当成单场景延迟计算加速比。
