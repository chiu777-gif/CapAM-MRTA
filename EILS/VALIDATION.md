# 实现验证

本文保留已执行的验证记录。日常操作见 [中文使用说明书](README.zh-CN.md)，查找正式结果见 [结果目录索引](results/README.md)。

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
