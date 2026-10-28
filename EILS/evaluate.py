"""Evaluate paper ILS/EILS profiles on fixed MRTA data; all output stays in EILS."""
import argparse
from collections import defaultdict
from contextlib import contextmanager, nullcontext
import csv
from datetime import datetime, timezone
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from EILS.navigation import load_evaluation_data, plan_scenario, summarize_planning
from z_test_path.path_planner import make_planner, planner_settings
from z_test_path.visibility_planner import DEFAULT_VISIBILITY_EPSILON
try:
    from .data_identity import dataset_identity
    from .solver import PROFILES, build, library, prepare, sha256, solve, validate_solution
except ImportError:
    from data_identity import dataset_identity
    from solver import PROFILES, build, library, prepare, sha256, solve, validate_solution

_worker_needs_warmup = False


def summarize(rows):
    total = sum(row["task_count"] for row in rows)
    done = sum(row["tasks_done"] for row in rows)
    durations = np.array([row["inference_seconds"] for row in rows])
    return {
        "scenarios": len(rows), "completed_tasks": done, "total_tasks": total,
        "completion_rate": done / total, "completion_percent": 100.0 * done / total,
        # imap_unordered changes summation order; keep serial/parallel means identical.
        "mean_cost": math.fsum(row["cost"] for row in rows) / len(rows),
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


def initialize_worker():
    global _worker_needs_warmup
    # The parent handles Ctrl+C and terminates the pool on interruption.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    torch.set_num_threads(1)
    _worker_needs_warmup = True


@contextmanager
def worker_thread_limits():
    """Limit native threads before spawned workers import NumPy/PyTorch."""
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
    original = {name: os.environ.get(name) for name in names}
    try:
        os.environ.update({name: "1" for name in names})
        yield
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def evaluate_scenario(job):
    """Solve one scenario's configurations; only the parent writes result files."""
    global _worker_needs_warmup
    data = job["data"]
    if _worker_needs_warmup:
        # Keep imports/library loading and warmup outside each solve's timers.
        # Run as a task so a warmup exception is propagated to the parent.
        solve(data, limit=1, seed=job["seed"], backend=job["backend"], service_time=job["service_time"],
              objective=job.get("objective", "completion"))
        _worker_needs_warmup = False
    seed = (job["seed"] + job["scenario_index"] + job["repeat"] * 1000003) % 2**64
    records = []
    for profile, algorithm in job["combinations"]:
        cpu_start = time.process_time()
        start = time.perf_counter()
        result = solve(data, algorithm=algorithm, profile=profile, seed=seed,
                       service_time=job["service_time"], limit=job["limit"], backend=job["backend"],
                       objective=job.get("objective", "completion"))
        duration = time.perf_counter() - start
        cpu_duration = time.process_time() - cpu_start
        metrics = validate_solution(data, result, job["service_time"])
        row = dict(backend=job["backend"], algorithm=algorithm, profile=profile, repeat=job["repeat"],
                   scenario_index=job["scenario_index"], instance=job["instance"],
                   deadline_group=job["deadline_group"],
                   seed=seed, task_count=data["n"], n_agents=data["m"],
                   **metrics, completion_percent=100.0 * metrics["tasks_done"] / data["n"],
                   native_completed_tasks=result["native_completed_tasks"],
                   iterations=result["iterations"], best_iteration=result["best_iteration"],
                   inference_seconds=duration, cpu_seconds=cpu_duration)
        row.update(path_planner=job["path_planner"], navigation_key=job["navigation_key"],
                   objective=job.get("objective", "completion"))
        route_record = dict(algorithm=algorithm, profile=profile, repeat=job["repeat"],
                            scenario_index=job["scenario_index"], routes=result["routes"],
                            finish_times=result["finish_times"].tolist(),
                            path_planner=job["path_planner"], navigation_key=job["navigation_key"],
                            objective=job.get("objective", "completion"))
        records.append((row, route_record))
    return records


def scenario_jobs(prepared, names, input_groups, opts, navigation_keys):
    # Alternate algorithm/profile order between scenarios to reduce ordering effects.
    combinations = [(profile, algorithm) for profile in opts.profiles for algorithm in opts.algorithms]
    for repeat in range(opts.repeats):
        for index, data in enumerate(prepared):
            selected = [pair for pair in combinations
                        if not opts.match_taptc_groups or pair[0] == input_groups[index]]
            yield dict(data=data, instance=names[index], deadline_group=input_groups[index],
                       scenario_index=opts.offset + index, repeat=repeat, seed=opts.seed,
                       combinations=selected if index % 2 == 0 else selected[::-1],
                       backend=opts.backend, service_time=opts.service_time, limit=opts.limit,
                       objective=getattr(opts, "objective", "completion"),
                       path_planner=opts.path_planner, navigation_key=navigation_keys[index])


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
    parser.add_argument("--objective", choices=["completion", "cost"], default="completion",
                        help="Maximize on-time task count, or minimize cost after completing every candidate")
    parser.add_argument("--match_taptc_groups", action="store_true",
                        help="Run only tight on group1 and slack on group2, once per algorithm/scenario")
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--service_time", choices=["continuous", "ceil"], default="continuous")
    parser.add_argument("--limit", type=int, help="Override iterations/trials, for verification only")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of CPU worker processes; default 1 preserves serial evaluation")
    parser.add_argument("--path_planner", choices=["auto", "euclidean", "visibility_dijkstra"], default="auto",
                        help="auto uses visibility graph + Dijkstra for obstacle datasets, Euclidean otherwise")
    parser.add_argument("--cache_dir", type=Path, default=REPO / "z_test_path/cache",
                        help="Navigation cache shared with z_test_path; independent of output_dir")
    parser.add_argument("--visibility_epsilon", type=float, default=DEFAULT_VISIBILITY_EPSILON)
    parser.add_argument("--path_workers", type=int, default=1,
                        help="CPU processes for the separate path precomputation phase")
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--progress_every", type=int, default=25,
                        help="Log after the first completed scenario, then every N scenarios; 0 disables progress")
    opts = parser.parse_args(args)
    if opts.repeats < 1 or opts.offset < 0 or not 0 <= opts.seed < 2**64:
        parser.error("Invalid repeats, offset or seed")
    if opts.num_samples is not None and opts.num_samples < 1:
        parser.error("num_samples must be positive")
    if opts.limit is not None and opts.limit < 1:
        parser.error("limit must be positive")
    if opts.workers < 1:
        parser.error("workers must be positive")
    if opts.path_workers < 1:
        parser.error("path_workers must be positive")
    if opts.progress_every < 0:
        parser.error("progress_every must be nonnegative")
    output = (opts.output_dir or HERE / "results" / ("n100_train_matched_1000_" + opts.backend)).resolve()
    if output != HERE and HERE not in output.parents:
        parser.error("output_dir must be inside EILS, as requested")
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is not empty: {}".format(output))
    if opts.progress_every:
        print("Loading dataset: {}".format(opts.data.resolve()), flush=True)
    torch.set_num_threads(1)
    dataset, obstacle_map, extent = load_evaluation_data(opts.data, opts.offset, opts.num_samples)
    if not len(dataset):
        parser.error("No scenarios selected")
    if opts.path_planner == "auto":
        opts.path_planner = "visibility_dijkstra" if obstacle_map is not None else "euclidean"
    if (obstacle_map is not None) != (opts.path_planner == "visibility_dijkstra"):
        parser.error("Obstacle datasets require visibility_dijkstra; visibility_dijkstra requires an obstacle map")
    if obstacle_map is not None:
        # Validate epsilon even when every scenario hits the disk cache.
        make_planner(obstacle_map, path_planner=opts.path_planner, visibility_epsilon=opts.visibility_epsilon)
    input_groups = [taptc_group(name) for name in dataset.names]
    if opts.match_taptc_groups and ("" in input_groups or not set(input_groups).issubset(opts.profiles)):
        parser.error("--match_taptc_groups requires original TAPTC cases and their matching --profiles")
    output.mkdir(parents=True, exist_ok=True)
    navigation_keys = [""] * len(dataset)
    navigations = [None] * len(dataset)
    planning_rows = []
    path_workers = min(opts.path_workers, len(dataset)) if obstacle_map is not None else 0
    planning_started = time.perf_counter()
    if obstacle_map is not None:
        if opts.progress_every:
            print("Precomputing visibility_dijkstra paths: {} scenarios, path_workers={}, cache={}".format(
                len(dataset), path_workers, opts.cache_dir.resolve()), flush=True)
        jobs = ((opts.offset + index,
                 {name: instance[name].cpu().numpy() for name in ("loc", "robots_start_location", "n_agents")},
                 obstacle_map, extent, opts.cache_dir.resolve(),
                 opts.visibility_epsilon, opts.rebuild_cache) for index, instance in enumerate(dataset))
        with (worker_thread_limits() if path_workers > 1 else nullcontext()), \
                (multiprocessing.get_context("spawn").Pool(path_workers, initializer=initialize_worker)
                 if path_workers > 1 else nullcontext(None)) as pool:
            results = pool.imap_unordered(plan_scenario, jobs, chunksize=1) if pool else map(plan_scenario, jobs)
            for completed, (index, tables, row) in enumerate(results, 1):
                navigations[index - opts.offset] = tables
                navigation_keys[index - opts.offset] = row["navigation_key"]
                planning_rows.append(row)
                if opts.progress_every and (completed == 1 or completed % opts.progress_every == 0
                                            or completed == len(dataset)):
                    print("Paths {}/{} scenarios; cache hits {}".format(
                        completed, len(dataset), sum(item["cache_hit"] for item in planning_rows)), flush=True)
        planning_rows.sort(key=lambda row: row["scenario_index"])
        write_csv(output / "path_planning.csv", planning_rows)
    planning_wall = time.perf_counter() - planning_started if obstacle_map is not None else 0.
    prepared = [prepare(instance, navigation) for instance, navigation in zip(dataset, navigations)]
    navigation_config = dict(path_planner=opts.path_planner, cache_dir=str(opts.cache_dir.resolve()),
                             visibility_epsilon=opts.visibility_epsilon if obstacle_map is not None else None,
                             path_workers=path_workers, path_workers_requested=opts.path_workers,
                             rebuild_cache=opts.rebuild_cache, navigation_keys=navigation_keys)
    if obstacle_map is not None:
        snapshot = output / "path_distances.npz"
        np.savez_compressed(snapshot, **{"{}_{}".format(name, index): data[name]
                            for index, data in enumerate(prepared)
                            for name in ("task_to_task", "robot_to_task")})
        navigation_config.update(planner=planner_settings(opts.path_planner, visibility_epsilon=opts.visibility_epsilon)["planner"],
                                 path_distances_file=snapshot.name, path_distances_sha256=sha256(snapshot))
    else:
        navigation_config["planner"] = "Euclidean distance"
    total_scenarios = len(prepared) * opts.repeats
    workers = min(opts.workers, total_scenarios)
    identity = dataset_identity(opts.data)
    binary = None
    if opts.backend == "cpp":
        binary = build()
        library()
    # Exclude imports/compilation, data loading/conversion and one warmup from timing.
    solve(prepared[0], limit=1, seed=opts.seed, backend=opts.backend, service_time=opts.service_time,
          objective=opts.objective)
    try:
        cpu_model = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                         if line.startswith("model name"))
    except (OSError, StopIteration):
        cpu_model = platform.processor()
    implementation = {"backend": opts.backend,
                      "solver_py_sha256": sha256(HERE / "solver.py"),
                      "evaluate_py_sha256": sha256(HERE / "evaluate.py"),
                      "data_identity_py_sha256": sha256(HERE / "data_identity.py")}
    implementation["navigation_py_sha256"] = sha256(HERE / "navigation.py")
    if opts.backend == "cpp":
        implementation.update(solver_cpp_sha256=sha256(HERE / "solver.cpp"),
                              binary_sha256=sha256(binary),
                              compiler=subprocess.check_output(["g++", "--version"], text=True).splitlines()[0],
                              flags="-O3 -std=c++17 -shared -fPIC (no fast-math)")
    else:
        implementation.update(python_solver_sha256=sha256(HERE / "python_solver.py"),
                              search_implementation="Python control and route operations; NumPy insertion candidates; no compiled solver extension")
    config = {
        **navigation_config,
        **summarize_planning(planning_rows),
        "path_planning_wall_seconds": planning_wall,
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
        "objective": opts.objective,
        "search_objective": ("minimize full-coverage cost after residual completion of each candidate; strict improvement; first best on ties"
                             if opts.objective == "cost" else
                             "maximize number of deadline-feasible tasks; strict improvement; first best on ties"),
        "search_neighborhood": "deadline-feasible insertion and original ILS/EILS perturbations; cost mode scores completed candidates without changing feasible search state",
        "residual_policy": "append unselected tasks to earliest-free robot using minimum travel+service time",
        "cost": "sum(finish_time/deadline for every task with finish_time > deadline), after residual completion",
        "fidelity": "paper-algorithm reimplementation with documented continuous-time and full-coverage adapters; not author-code replication",
        "capam_baseline_parameters_published": False,
        "timing": {
            "device": "CPU", "workers": workers, "workers_requested": opts.workers,
            "native_threads": 1, "batch_size": 1,
            "parallel_unit": "scenario", "process_start_method": "spawn" if workers > 1 else None,
            "warmup_calls": 1,
            "warmup_scope": "parent", "worker_warmup_calls_per_process": 1 if workers > 1 else 0,
            "includes": "{} solve wrapper, pairwise travel/service preparation, entire search, residual completion, route extraction".format(opts.backend),
            "excludes": "path planning, cache I/O, compilation, library/model loading, dataset I/O, tensor conversion, warmup, independent route replay, metrics, process startup, IPC and result I/O",
            "path_planning_seconds_total": "Sum of actual graph construction, Dijkstra searches and distance-table computation per selected scenario, excluding cache I/O; cache hits contribute zero; counted once across all algorithms/profiles/repeats",
            "path_planning_wall_seconds": "Separate precomputation phase wall time, including cache I/O and planning worker startup/IPC; excludes search and distance snapshot export",
            "clock": "perf_counter wall time; process_time also reported",
            "evaluation_wall_includes": "evaluation, route replay, result I/O, and parallel pool startup/shutdown and IPC",
            "latency_context": "serial" if workers == 1 else "concurrent CPU workers; per-solve latency includes resource contention",
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
    if opts.progress_every:
        print("Starting evaluation: {} scenarios, {} runs, backend={}, workers={}; results: {}".format(
            total_scenarios, total_runs, opts.backend, workers, output), flush=True)
    with (output / "scenarios.csv").open("w", newline="", encoding="utf-8") as scenarios, \
            (output / "routes.jsonl").open("w", encoding="utf-8") as routes_file, \
            (worker_thread_limits() if workers > 1 else nullcontext()) as thread_limits, \
            (multiprocessing.get_context("spawn").Pool(workers, initializer=initialize_worker)
             if workers > 1 else nullcontext(None)) as pool:
        writer = None
        jobs = scenario_jobs(prepared, dataset.names, input_groups, opts, navigation_keys)
        results = pool.imap_unordered(evaluate_scenario, jobs, chunksize=1) if pool else map(evaluate_scenario, jobs)
        for completed, records in enumerate(results, 1):
            for row, route_record in records:
                if writer is None:
                    writer = csv.DictWriter(scenarios, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                routes_file.write(json.dumps(route_record, separators=(",", ":")) + "\n")
                rows.append(row)
            # Keep partial results visible even when progress logging is disabled.
            scenarios.flush()
            routes_file.flush()
            if opts.progress_every and (completed == 1 or completed % opts.progress_every == 0
                                        or completed == total_scenarios):
                elapsed = time.perf_counter() - started
                print("{}/{} scenarios, {}/{} runs, {:.1f}s elapsed, {:.1f}s estimated remaining".format(
                    completed, total_scenarios, len(rows), total_runs, elapsed,
                    elapsed * (total_runs - len(rows)) / len(rows)), flush=True)
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
              "优化目标：`{}`（{}）。".format(opts.objective,
                  "最小化每轮候选方案补全全部任务后的 cost" if opts.objective == "cost" else "最大化按时完成任务数"), "",
              "距离：`{}`（{}）；推理全程使用预计算距离表。".format(opts.path_planner, navigation_config["planner"])
                  if obstacle_map is not None else "距离：欧几里得距离。", "",
              "路径规划累计时间：{:.6f} 秒；规划阶段墙钟时间：{:.6f} 秒；缓存命中 {}/{}。".format(
                  config["path_planning_seconds_total"], planning_wall, config["cache_hits"], len(planning_rows)), "",
              "路径规划与缓存读写不计入推理时间；每个场景只预计算一次，供所有算法、参数组和重复运行复用。", "",
              "CPU 工作进程数：{}；每个进程逐场景求解。".format(workers), "",
              ("时间为单 CPU 线程、逐场景完整求解的平均延迟。" if workers == 1 else
               "时间为多个 CPU 进程并行运行时，单次完整求解的平均延迟，可能受资源竞争影响；不是整批耗时除以求解次数。"), "",
              "整批评估墙钟时间：{:.3f} 秒；各次推理时间之和在并行时可大于整批墙钟时间。".format(
                  summary["evaluation_wall_seconds"]), "",
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
    print(json.dumps(dict(path_planner=opts.path_planner, **summarize_planning(planning_rows),
                          path_planning_wall_seconds=planning_wall), indent=2))
    print("Results: {}".format(output), flush=True)
    return summary


if __name__ == "__main__":
    main()
