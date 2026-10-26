"""Evaluate paper ILS/EILS profiles on fixed MRTA data; all output stays in EILS."""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from problems.mrta.data import load_file_dataset
try:
    from .data_identity import dataset_identity
    from .solver import PROFILES, build, library, prepare, sha256, solve, validate_solution
except ImportError:
    from data_identity import dataset_identity
    from solver import PROFILES, build, library, prepare, sha256, solve, validate_solution


def summarize(rows):
    total = sum(row["task_count"] for row in rows)
    done = sum(row["tasks_done"] for row in rows)
    durations = np.array([row["inference_seconds"] for row in rows])
    return {
        "scenarios": len(rows), "completed_tasks": done, "total_tasks": total,
        "completion_rate": done / total, "completion_percent": 100.0 * done / total,
        "mean_cost": float(np.mean([row["cost"] for row in rows])),
        "inference_seconds_total": float(durations.sum()),
        "inference_seconds_per_scenario": float(durations.mean()),
        "inference_seconds_p50": float(np.median(durations)),
        "inference_seconds_p95": float(np.quantile(durations, 0.95)),
        "cpu_seconds_per_scenario": float(np.mean([row["cpu_seconds"] for row in rows])),
        "mean_iterations": float(np.mean([row["iterations"] for row in rows])),
    }


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def taptc_group(name):
    match = re.fullmatch(r"r([12])[1-4]a\d+i\d+", Path(name).stem)
    return ("tight" if match.group(1) == "1" else "slack") if match else ""


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=REPO / "z_test/data/n100_train_matched_1000_seed20261024.pkl")
    parser.add_argument("--backend", choices=["python", "cpp"], default="python",
                        help="Python/NumPy search by default; cpp requires g++")
    parser.add_argument("--output_dir", type=Path,
                        help="Default: EILS/results/n100_train_matched_1000_<backend>")
    parser.add_argument("--algorithms", choices=["ils", "eils"], nargs="+", default=["ils", "eils"])
    parser.add_argument("--profiles", choices=list(PROFILES), nargs="+", default=list(PROFILES))
    parser.add_argument("--match_taptc_groups", action="store_true",
                        help="Run only tight on group1 and slack on group2, once per algorithm/scenario")
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--service_time", choices=["continuous", "ceil"], default="continuous")
    parser.add_argument("--limit", type=int, help="Override iterations/trials, for verification only")
    parser.add_argument("--progress_every", type=int, default=25)
    opts = parser.parse_args(args)
    if opts.repeats < 1 or opts.offset < 0 or not 0 <= opts.seed < 2**64:
        parser.error("Invalid repeats, offset or seed")
    if opts.num_samples is not None and opts.num_samples < 1:
        parser.error("num_samples must be positive")
    if opts.limit is not None and opts.limit < 1:
        parser.error("limit must be positive")
    output = (opts.output_dir or HERE / "results" / ("n100_train_matched_1000_" + opts.backend)).resolve()
    if output != HERE and HERE not in output.parents:
        parser.error("output_dir must be inside EILS, as requested")
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is not empty: {}".format(output))
    torch.set_num_threads(1)
    dataset = load_file_dataset([opts.data.resolve()], offset=opts.offset, num_samples=opts.num_samples)
    prepared = [prepare(instance) for instance in dataset]
    input_groups = [taptc_group(name) for name in dataset.names]
    if opts.match_taptc_groups and ("" in input_groups or not set(input_groups).issubset(opts.profiles)):
        parser.error("--match_taptc_groups requires original TAPTC cases and their matching --profiles")
    identity = dataset_identity(opts.data)
    binary = None
    if opts.backend == "cpp":
        binary = build()
        library()
    # Exclude imports/compilation, data loading/conversion and one warmup from timing.
    solve(prepared[0], limit=1, seed=opts.seed, backend=opts.backend, service_time=opts.service_time)
    output.mkdir(parents=True, exist_ok=True)
    try:
        cpu_model = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                         if line.startswith("model name"))
    except (OSError, StopIteration):
        cpu_model = platform.processor()
    implementation = {"backend": opts.backend,
                      "solver_py_sha256": sha256(HERE / "solver.py"),
                      "evaluate_py_sha256": sha256(HERE / "evaluate.py"),
                      "data_identity_py_sha256": sha256(HERE / "data_identity.py")}
    if opts.backend == "cpp":
        implementation.update(solver_cpp_sha256=sha256(HERE / "solver.cpp"),
                              binary_sha256=sha256(binary),
                              compiler=subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
                              flags="-O3 -std=c++17 -shared -fPIC (no fast-math)")
    else:
        implementation.update(python_solver_sha256=sha256(HERE / "python_solver.py"),
                              search_implementation="Python control and route operations; NumPy insertion candidates; no compiled solver extension")
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend": opts.backend,
        "dataset": str(opts.data.resolve()), "dataset_sha256": identity["sha256"],
        "dataset_identity": identity,
        "scenarios_per_run": len(dataset), "offset": opts.offset,
        "robot_count_histogram": {str(m): sum(data["m"] == m for data in prepared)
                                  for m in sorted({data["m"] for data in prepared})},
        "algorithms": opts.algorithms, "profiles": {name: PROFILES[name] for name in opts.profiles},
        "profile_mode": "taptc_matched" if opts.match_taptc_groups else "fixed",
        "limit_override": opts.limit, "repeats": opts.repeats, "seed": opts.seed,
        "scenario_seed_rule": "(seed + global_scenario_index + repeat * 1000003) modulo 2**64",
        "random_generator": "MT19937-64; raw uint64 output modulo tour length",
        "service_time": opts.service_time,
        "search_objective": "maximize number of deadline-feasible tasks; strict improvement; first best on ties",
        "residual_policy": "append unselected tasks to earliest-free robot using minimum travel+service time",
        "cost": "sum(finish_time/deadline for every task with finish_time > deadline), after residual completion",
        "fidelity": "paper-algorithm reimplementation with documented continuous-time and full-coverage adapters; not author-code replication",
        "capam_baseline_parameters_published": False,
        "timing": {
            "device": "CPU", "workers": 1, "native_threads": 1, "batch_size": 1,
            "warmup_calls": 1,
            "includes": "{} solve wrapper, pairwise travel/service preparation, entire search, residual completion, route extraction".format(opts.backend),
            "excludes": "compilation, library/model loading, dataset I/O, tensor conversion, warmup, independent route replay, metrics and result I/O",
            "clock": "perf_counter wall time; process_time also reported",
        },
        "hardware": {"cpu": cpu_model, "logical_cpus": os.cpu_count(),
                     "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                     "platform": platform.platform(), "python": platform.python_version(),
                     "numpy": np.__version__, "torch": torch.__version__},
        "implementation": implementation,
    }
    (output / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    rows = []
    started = time.perf_counter()
    total_runs = len(prepared) * len(opts.algorithms) * (1 if opts.match_taptc_groups else len(opts.profiles)) * opts.repeats
    with (output / "scenarios.csv").open("w", newline="", encoding="utf-8") as scenarios, \
            (output / "routes.jsonl").open("w", encoding="utf-8") as routes_file:
        writer = None
        # Alternate algorithm/profile order between scenarios to reduce ordering effects.
        combinations = [(profile, algorithm) for profile in opts.profiles for algorithm in opts.algorithms]
        for repeat in range(opts.repeats):
            for index, data in enumerate(prepared):
                global_index = opts.offset + index
                seed = (opts.seed + global_index + repeat * 1000003) % 2**64
                selected_combinations = [pair for pair in combinations
                                         if not opts.match_taptc_groups or pair[0] == input_groups[index]]
                for profile, algorithm in (selected_combinations if index % 2 == 0 else selected_combinations[::-1]):
                    cpu_start = time.process_time()
                    start = time.perf_counter()
                    result = solve(data, algorithm=algorithm, profile=profile, seed=seed,
                                   service_time=opts.service_time, limit=opts.limit, backend=opts.backend)
                    duration = time.perf_counter() - start
                    cpu_duration = time.process_time() - cpu_start
                    metrics = validate_solution(data, result, opts.service_time)
                    row = dict(backend=opts.backend, algorithm=algorithm, profile=profile, repeat=repeat,
                               scenario_index=global_index, instance=dataset.names[index],
                               deadline_group=input_groups[index],
                               seed=seed, task_count=data["n"], n_agents=data["m"],
                               **metrics, completion_percent=100.0 * metrics["tasks_done"] / data["n"],
                               native_completed_tasks=result["native_completed_tasks"],
                               iterations=result["iterations"], best_iteration=result["best_iteration"],
                               inference_seconds=duration, cpu_seconds=cpu_duration)
                    if writer is None:
                        writer = csv.DictWriter(scenarios, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    route_record = dict(algorithm=algorithm, profile=profile, repeat=repeat,
                                        scenario_index=global_index, routes=result["routes"],
                                        finish_times=result["finish_times"].tolist())
                    routes_file.write(json.dumps(route_record, separators=(",", ":")) + "\n")
                    rows.append(row)
                if opts.progress_every and ((index + 1) % opts.progress_every == 0 or index + 1 == len(prepared)):
                    elapsed = time.perf_counter() - started
                    print("{}/{} runs, {:.1f}s elapsed, {:.1f}s estimated remaining".format(
                        len(rows), total_runs, elapsed, elapsed * (total_runs - len(rows)) / len(rows)), flush=True)
                    scenarios.flush()
                    routes_file.flush()
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["algorithm"], row["profile"], row["n_agents"])].append(row)
    groups = [dict(algorithm=key[0], profile=key[1], n_agents=key[2], **summarize(items))
              for key, items in sorted(grouped.items())]
    write_csv(output / "groups.csv", groups)
    models = {}
    for profile in opts.profiles:
        for algorithm in opts.algorithms:
            subset = [row for row in rows if row["algorithm"] == algorithm and row["profile"] == profile]
            if not subset:
                continue
            models[algorithm + "_" + profile] = dict(
                **summarize(subset), by_robot_count={str(group["n_agents"]): group for group in groups
                    if group["algorithm"] == algorithm and group["profile"] == profile})
    by_deadline_group = []
    for (algorithm, profile, m), items in sorted(grouped.items()):
        for deadline_group in ("tight", "slack"):
            subset = [row for row in items if row["deadline_group"] == deadline_group]
            if subset:
                by_deadline_group.append(dict(algorithm=algorithm, profile=profile, n_agents=m,
                                              deadline_group=deadline_group, **summarize(subset)))
    if by_deadline_group:
        write_csv(output / "deadline_groups.csv", by_deadline_group)
    matched_models = {}
    for algorithm in opts.algorithms:
        subset = [row for row in rows if row["algorithm"] == algorithm and row["profile"] == row["deadline_group"]]
        expected = len(prepared) * opts.repeats
        if len(subset) == expected:
            matched_models[algorithm] = dict(
                **summarize(subset),
                selection_rule="group1 -> tight parameters, group2 -> slack parameters; selected by input group, not outcome",
                by_robot_count={str(m): summarize([row for row in subset if row["n_agents"] == m])
                                for m in sorted({row["n_agents"] for row in subset})},
                by_deadline_group={group: summarize([row for row in subset if row["deadline_group"] == group])
                                   for group in sorted({row["deadline_group"] for row in subset})})
    summary = dict(config, evaluation_wall_seconds=time.perf_counter() - started,
                   independent_route_replay="passed for every solution",
                   models=models, matched_models=matched_models, by_deadline_group=by_deadline_group)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = ["# ILS / EILS 固定数据集测试", "",
              "数据：`{}`，共 {} 个原始场景。".format(opts.data.resolve(), len(dataset)), "",
              "求解后端：`{}`。".format(opts.backend), "",
              "时间为单 CPU 线程、逐场景完整求解的平均延迟。", "",
              "算法细节、复现边界和 cost 补全规则见 [SOURCES.md](../../SOURCES.md)。", ""]
    if matched_models:
        report.extend(["## 原始 TAPTC 分组对应参数", "",
                       "根据输入分组选择参数：group1 使用 tight，group2 使用 slack；不根据测试结果选参。", "",
                       "| 算法 | 场景数 | 平均 cost | 按时完成率 | 平均推理时间 (ms) | 总推理时间 (s) |",
                       "|---|---:|---:|---:|---:|---:|"])
        for algorithm, model in matched_models.items():
            report.append("| {} | {} | {:.6f} | {:.3f}% | {:.3f} | {:.6f} |".format(
                algorithm.upper(), model["scenarios"], model["mean_cost"], model["completion_percent"],
                1000 * model["inference_seconds_per_scenario"], model["inference_seconds_total"]))
        report.extend(["", "| 算法 | 机器人 | 场景数 | 平均 cost | 按时完成率 | 平均推理时间 (ms) |",
                       "|---|---:|---:|---:|---:|---:|"])
        for algorithm, model in matched_models.items():
            for m, group in model["by_robot_count"].items():
                report.append("| {} | {} | {} | {:.6f} | {:.3f}% | {:.3f} |".format(
                    algorithm.upper(), m, group["scenarios"], group["mean_cost"], group["completion_percent"],
                    1000 * group["inference_seconds_per_scenario"]))
        report.extend(["", ("每种算法在上述 {} 个场景上各运行一次，按原始分组直接使用对应参数。".format(len(dataset))
                            if opts.match_taptc_groups else
                            "以上 {} 个场景的结果从下方固定配置中按原始分组提取，不是额外的一轮实验。".format(len(dataset))), ""])
    report.extend([("## 每套参数在对应输入分组上的结果" if opts.match_taptc_groups else "## 每套固定参数在全部场景上的结果"), "",
                   "| 参数组 | 算法 | 场景数 | 平均 cost | 按时完成率 | 平均推理时间 (ms) | 总推理时间 (s) |",
                   "|---|---|---:|---:|---:|---:|---:|"])
    for name, model in models.items():
        algorithm, profile = name.split("_", 1)
        report.append("| {} | {} | {} | {:.6f} | {:.3f}% | {:.3f} | {:.6f} |".format(
            profile, algorithm.upper(), model["scenarios"], model["mean_cost"], model["completion_percent"],
            1000 * model["inference_seconds_per_scenario"], model["inference_seconds_total"]))
    report.extend(["", "| 参数组 | 算法 | 机器人 | 场景数 | 平均 cost | 按时完成率 | 平均推理时间 (ms) |",
                   "|---|---|---:|---:|---:|---:|---:|"])
    for group in groups:
        report.append("| {} | {} | {} | {} | {:.6f} | {:.3f}% | {:.3f} |".format(
            group["profile"], group["algorithm"].upper(), group["n_agents"], group["scenarios"],
            group["mean_cost"], group["completion_percent"], 1000 * group["inference_seconds_per_scenario"]))
    report.extend(["", "cost 包含原始可行路线之外任务的统一补全；完成率等于原始搜索得分。",
                   ("连续服务时间与当前 CapAM 环境一致。" if opts.service_time == "continuous"
                    else "服务时间使用 ceil(workload/capacity)，与原始 TAPTC 文献定义一致。"),
                   ("每个场景仅使用其原始分组对应的参数，不按测试结果选参。" if opts.match_taptc_groups else
                    "每套指定的参数组都在全部场景上完整测试，不按测试结果选参。"),
                   "当前每个配置每场景重复 {} 次，原始样本数为 {}。".format(opts.repeats, len(dataset)), ""])
    (output / "REPORT.md").write_text("\n".join(report))
    print(json.dumps({name: {key: value for key, value in model.items() if key != "by_robot_count"}
                      for name, model in models.items()}, indent=2))
    print("Results: {}".format(output), flush=True)
    return summary


if __name__ == "__main__":
    main()
