# 实现验证

本文保留已执行的验证记录。日常操作见 [中文使用说明书](README.zh-CN.md)，查找正式结果见 [结果目录索引](results/README.md)。

## 可配置优化目标

- 新增 `--objective completion|cost` 和公共 `solve(..., objective=...)`；默认 completion 的路线与搜索统计回归测试通过。
- `python -m unittest EILS.test_objective`：5 项测试通过，覆盖完整补全 cost 参与最好解选择、评分后恢复搜索状态、零 cost 下界提前退出、Python/C++ 在两种服务时间与有无预计算距离时的一致性，以及 cost 目标串并行一致性和落盘目标篡改检查。
- 原有求解、Python 后端、障碍导航及并行评估测试均通过；`g++ -std=c++17 -Wall -Wextra -Wpedantic -fsyntax-only EILS/solver.cpp` 通过。
- cost 模式保留原有可行插入/扰动邻域，使用完整补全方案的 cost 做严格改进判断；不是任意完整路线邻域或全局最优算法。详细定义见使用说明书的“切换优化目标”。
- 指定障碍数据 `z_test_path/data/n100_obstacles_example.pkl` 的全部 100 场景，Python、ILS/EILS × tight/slack，两个目标共 800 次完整预算求解全部完成，分别通过 `verify_results.py` 的 400 条独立重放。seed=6732、epsilon=0.01、path_workers=8、workers=8；目标之间顺序运行，无磁盘路径缓存，沿用前次实验的运行时双向碰撞检查。两轮预计算距离矩阵逐项完全一致，记录的求解/评估源码哈希与运行后的文件一致。
- 结果见 [优化目标对照报告](../final_data/ils_eils_objective_comparison_0.01/REPORT.md)：切换为 cost 目标后，四套配置的平均 cost 下降约 6.92%～10.28%，完成率下降 0.75～1.32 个百分点。该结论限定于本次数据、种子和停止规则。

## 原始实现与初次评估

2026-10-25，在项目 `mrta_c` Python 环境中完成。

- `python -m unittest EILS.test_solver -v`：9 项测试通过。构造步骤与独立穷举插入参考实现核对 30 组随机小问题；另覆盖截止等号、不可行任务、空路线、100% 完成提前停止、连续/取整服务时间、归一化与非活跃机器人、固定种子复现、公共模拟器一致性和非法输入。
- `g++ -std=c++17 -Wall -Wextra -Wpedantic -fsyntax-only EILS/solver.cpp`：通过，无编译警告。
- `g++ -O1 -g -std=c++17 -fsanitize=address,undefined EILS/test_native.cpp -o EILS/build/test_native`，随后 `ASAN_OPTIONS=detect_leaks=0 EILS/build/test_native`：通过。覆盖循环删除回绕、整条路线删除、空路线、删除后时间/最大允许延迟，以及 100 次随机起点删除。容器 ptrace 环境不支持 LeakSanitizer，故关闭泄漏检测；地址和未定义行为检测保持开启。
- `python -m py_compile EILS/solver.py EILS/evaluate.py EILS/test_solver.py`：通过。
- 小样本评估：前 5 个场景、4 个配置，共 20 次完整求解，全部通过即时独立路线重放。
- `python EILS/verify_results.py EILS/results/pilot`：重新读取 20 个保存解和 12 个已有分组，验证通过。
- 正式全量评估：1000 个场景 × ILS/EILS × tight/slack，共 4000 次求解，544.485 秒结束。全部求解即时独立路线重放通过。
- `python EILS/verify_results.py EILS/results/n100_train_matched_1000`：重新读取全部 4000 个落盘解，24 个分组及 4 个总体汇总核对通过。原始数据 SHA256 为 `cd11acb2ea58525f99f7bdcd6cef862e2c4ecc31676c6b8362fd3d2403f97bc9`。

正式实验在 `EILS/results/n100_train_matched_1000/`。其 `verification.json` 是全量完成后重新读取实际落盘数据的审计结果；它核验所有配置/场景组合、路线、任务完成时间、cost、完成率、随机种子和分组统计。实际测试日志在 `EILS/full_evaluation.log`。

验证范围是本实现的正确性和结果可复算性，不代表获得了作者未公开代码，也不消除 [SOURCES.md](SOURCES.md) 中明确记录的复现边界。

## 原始 TAPTC 96 场景评估

- 新增目录数据来源校验，`python -m unittest EILS.test_data_identity -v` 的 3 项测试通过：保留单文件哈希兼容性，感知任务/机器人数据变化，忽略未读取的扩展数据，缺少机器人文件时拒绝运行。
- 输入为 `data/mrta_gini/group1`、`group2` 的 96 个任务文件和对应的 12 个机器人文件，每场景 100 个任务；机器人数量 2/3/5/7 各 24 个场景。
- 运行 ILS/EILS × tight/slack，共 384 次完整求解，评估循环用时 40.970 秒；服务时间为连续时间，cost 沿用统一剩余任务补全规则。
- `python EILS/verify_results.py EILS/results/taptc96` 通过：384 个保存解、16 个机器人分组、32 个真实截止时间分组、2 个原始分组匹配参数汇总全部一致。
- 目录数据指纹为 `4745b8a63bb2258ee3fc654e0d8c2c253a833fae2a823b285c50651149544c21`，完整清单保存在对应的 `run_config.json` / `summary.json`。
- 原始分组匹配汇总：group1 使用 tight 参数、group2 使用 slack 参数，配置由输入分组决定；每种算法汇总 96 个实际场景，不额外求解。
- 目录支持扩展后，旧格式小样本结果的 20 个保存解和 12 个分组仍通过复算兼容性检查。

## Python / NumPy 后端

- 默认后端切换为 `python`；`--backend cpp` 仍可使用原来的 C++ 实现。Python 求解与评估路径不调用编译器或加载原生求解库。
- 更新后原有 `EILS.test_solver` 和 `EILS.test_data_identity` 共 12 项测试通过；`EILS.test_python_backend` 另有 7 项测试通过，共 19 项。
- Python/C++ 单元核对覆盖 8 组随机问题 × 3 种构造/搜索模式 × 2 组参数 × 2 种服务时间，共 96 个配置；逐项比较路线、机器人分配、完成时间和搜索统计。
- MT19937-64 参考值由 C++ 标准引擎生成；核对种子 0、5489、`2**64-1` 在输出序号 0、1、311、312、623、624 的结果，覆盖状态更新边界。
- 发现并修复 `math.hypot` 与 C++ `hypot` 的舍入差异：在 TAPTC `r11a5i01` 中，约 `10^-14` 量级的距离误差会改变接近并列的插入顺序。改为 NumPy `hypot` 后，当前环境的距离矩阵与 C++ 完全一致。该场景加入回归测试。
- 在原始 96 个 TAPTC 场景上，对每种算法执行短预算 Python/C++ 核对，共 192 个解，路线、完成时间、完成数、实际迭代数和最好解所在迭代数均一致。
- Python 评估入口在禁止调用构建/原生库函数的测试中运行成功，并能重读所有保存路线；另测试跨 group1/group2 的原始分组匹配模式，每个场景每种算法仅运行一次。
- 初次 Python 全量运行在发现上述浮点差异后中断，部分结果仅归档于 `results/taptc96_python_interrupted_hypot_check/`，其中 `STATUS.json` 明确标记不可作为最终基准。正式结果保存在 `results/taptc96_python/`。
- 语法编译检查通过：`python -m py_compile EILS/python_solver.py EILS/solver.py EILS/evaluate.py EILS/verify_results.py EILS/compare_backends.py EILS/test_python_backend.py`。
- Python 正式运行：原始 TAPTC 96 场景，group1 使用 tight、group2 使用 slack，每种算法每场景各一次，共 192 个解；评估循环总用时 758.489 秒。ILS 平均每场景 1.771309 秒，EILS 为 6.127080 秒。
- `python EILS/verify_results.py EILS/results/taptc96_python` 通过：192 个保存解、16 个机器人分组、16 个截止时间分组及 2 个匹配参数总体汇总全部复算一致。
- `python EILS/compare_backends.py EILS/results/taptc96 EILS/results/taptc96_python` 通过：192 个对应解的任务/机器人路线、完成数、cost、实际迭代数和最好解迭代数全部一致；完成时间最大绝对差为 **0**。
- 正式运行结束后核对 Python 求解器、公共接口、评估入口和数据身份模块的 SHA256，均与开跑时的记录一致。

## 多进程场景评估

2026-10-27，在 `mrta_c` 环境中完成。

- 新增 `--workers`，默认 `1`；使用独立进程并行处理场景，主进程按完成顺序统一写入 CSV 和路线文件。工作进程使用 `spawn`，限制原生计算线程，并在首次计时前单独预热。
- `python -m unittest discover -s EILS -t . -p 'test_*.py' -v`：23 项测试通过，用时 11.144 秒。新增 4 项测试覆盖固定参数下串行/并行一致性、跨 group1/group2 的匹配模式、Python/C++ 后端、连续/取整服务时间、非零 offset、重复运行、种子溢出取模、直接执行脚本的进程启动及进度输出、进程数上限和非法进程数。
- 串行与并行的路线、完成时间、完成率、cost、种子及实际/最好迭代数逐项一致。两种执行模式的落盘结果均通过 `verify_results.py` 独立复算；推理耗时和进程 CPU 时间分别记录，不要求时间数值一致。
- 对训练分布匹配的 1000 场景数据，取前 8 个场景运行 ILS/EILS × tight/slack，共 32 次求解，种子为 `20261025`，调试预算为 `--limit 300`。串行评估墙钟时间为 **41.843 秒**，4 进程为 **12.173 秒**，本次小样本加速约 **3.44 倍**；两次结果完全一致，各自 32 个保存解均复算通过。使用临时目录，验证后已清理。
- 上述加速测试是短预算小样本验证，不能视为正式预算下 1000 场景的耗时结果。并行墙钟时间包含进程池启动/关闭、工作进程预热、通信、求解、重放与结果输出；各次 `inference_seconds` 仍仅测量该次完整求解的延迟。
- `python -m py_compile EILS/evaluate.py EILS/test_parallel_evaluation.py` 及 `git diff --check` 通过。

## 障碍数据与导航缓存

2026-10-27，在 `mrta_c` 环境中完成。

- EILS 复用 `z_test_path` 的可见图 + Dijkstra 及缓存键规则。提前规划与搜索分为两个阶段，分别由 `--path_workers`、`--workers` 控制；Python/C++ 搜索、剩余任务补全与独立路线复算均读取相同的距离表。
- `python -m unittest discover -s EILS -t . -p 'test_*.py'`：28 项测试通过，用时 15.263 秒。新增 5 项测试覆盖：欧氏距离会使任务可行、绕行后不可行的对照；真实障碍折线的碰撞/距离检查；内部地图边长为 2 的尺度转换；忽略位于障碍内的非活跃机器人；两种后端和服务时间；规划/搜索串并行一致性；参数组与 repeats 共享预计算；epsilon 改变缓存键；强制重算；删除共享缓存后的独立复算；距离快照篡改检测；落盘后端比较。
- 并行预计算只向工作进程传递 NumPy 坐标，避免触发 PyTorch 张量跨进程共享所需的额外 socket。
- 使用 `math.fsum` 汇总 cost，避免并行完成顺序改变导致平均 cost 出现约 `10^-14` 的舍入差异；逐场景结果不变。
- `python -m unittest z_test_path.test_visibility_planner`：6 项测试通过。`g++ -std=c++17 -Wall -Wextra -Wpedantic -fsyntax-only EILS/solver.cpp` 和 `git diff --check` 通过。
- 真实数据冷缓存试跑：`n100_obstacles_example.pkl` 前 2 个场景，Python EILS-tight，`--limit 10`，串行规划与推理。实际规划累计 2.715583 秒，规划阶段墙钟 2.792830 秒，缓存写入累计 0.072596 秒；两个保存解复算通过。结果在 `results/n100_obstacles_visibility_cold_smoke/`，这是短预算验证，不作为正式算法效果或推理速度结果。
- 真实数据全量运行：100 场景 × EILS-tight/slack，共 200 次完整预算求解，Python 后端，4 个推理进程。100/100 命中已有导航缓存，实际规划计算为 0 秒，提前规划阶段墙钟 1.622656 秒。评估循环约 648.4 秒；tight 完成率 84.59%、平均 cost 72.721299，slack 完成率 83.16%、平均 cost 77.458222。结果在 `results/n100_obstacles_visibility_python_run01/`，逐场景即时独立重放全部通过；`verify_results.py` 重新读取全部 200 个保存解、12 个机器人分组和各套参数的总体汇总，复算通过。
