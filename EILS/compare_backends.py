"""Compare saved Python solutions with the matching C++ run, outside timing."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def key(row):
    return (row["algorithm"], row["profile"], int(row["repeat"]), int(row["scenario_index"]))


def read_rows(directory):
    with (directory / "scenarios.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    result = {key(row): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate scenario records in {}".format(directory))
    return result


def read_routes(directory):
    with (directory / "routes.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle]
    result = {key(row): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate route records in {}".format(directory))
    return result


def compare(cpp_directory, python_directory):
    cpp_directory, python_directory = Path(cpp_directory), Path(python_directory)
    cpp_summary = json.loads((cpp_directory / "summary.json").read_text())
    py_summary = json.loads((python_directory / "summary.json").read_text())
    if cpp_summary.get("backend", "cpp") != "cpp" or py_summary.get("backend") != "python":
        raise ValueError("Expected C++ and Python result directories, in that order")
    if cpp_summary.get("objective", "completion") != py_summary.get("objective", "completion"):
        raise ValueError("Experiment settings differ: objective")
    for field in ("dataset_sha256", "scenarios_per_run", "offset", "service_time", "seed", "repeats",
                  "limit_override", "profiles", "algorithms", "residual_policy", "search_objective"):
        if cpp_summary[field] != py_summary[field]:
            raise ValueError("Experiment settings differ: {}".format(field))
    if cpp_summary.get("path_planner", "euclidean") != py_summary.get("path_planner", "euclidean"):
        raise ValueError("Experiment settings differ: path_planner")
    if py_summary.get("path_planner") == "visibility_dijkstra":
        for field in ("visibility_epsilon", "navigation_keys"):
            if cpp_summary.get(field) != py_summary.get(field):
                raise ValueError("Experiment settings differ: {}".format(field))
    cpp_rows, py_rows = read_rows(cpp_directory), read_rows(python_directory)
    cpp_routes, py_routes = read_routes(cpp_directory), read_routes(python_directory)
    if set(py_rows) != set(py_routes) or not set(py_rows).issubset(cpp_rows) or not set(py_rows).issubset(cpp_routes):
        raise ValueError("Missing corresponding scenario/route results")
    mismatches = []
    max_finish_difference = 0.0
    for identifier, row in py_rows.items():
        reference = cpp_rows[identifier]
        differences = []
        for field in ("task_count", "n_agents", "seed", "tasks_done", "native_completed_tasks", "iterations", "best_iteration"):
            if int(row[field]) != int(reference[field]):
                differences.append(field)
        if not np.isclose(float(row["cost"]), float(reference["cost"]), atol=1e-9, rtol=1e-12):
            differences.append("cost")
        if py_routes[identifier]["routes"] != cpp_routes[identifier]["routes"]:
            differences.append("routes")
        finish = np.asarray(py_routes[identifier]["finish_times"])
        expected = np.asarray(cpp_routes[identifier]["finish_times"])
        if finish.shape != expected.shape:
            differences.append("finish_time_shape")
        else:
            max_finish_difference = max(max_finish_difference, float(np.abs(finish - expected).max()))
            if not np.allclose(finish, expected, atol=1e-9, rtol=1e-12):
                differences.append("finish_times")
        if differences:
            mismatches.append({"key": identifier, "fields": differences})
    timings = {}
    for algorithm in py_summary["algorithms"]:
        selected = [identifier for identifier in py_rows if identifier[0] == algorithm]
        cpp_seconds = sum(float(cpp_rows[identifier]["inference_seconds"]) for identifier in selected)
        py_seconds = sum(float(py_rows[identifier]["inference_seconds"]) for identifier in selected)
        timings[algorithm] = {
            "scenarios": len(selected), "cpp_seconds_per_scenario": cpp_seconds / len(selected),
            "python_seconds_per_scenario": py_seconds / len(selected), "python_seconds_total": py_seconds,
            "python_over_cpp_time_ratio": py_seconds / cpp_seconds,
            "completion_percent": float(np.mean([float(py_rows[identifier]["completion_percent"]) for identifier in selected])),
            "mean_cost": float(np.mean([float(py_rows[identifier]["cost"]) for identifier in selected])),
        }
    result = {"status": "passed" if not mismatches else "failed",
              "solutions_compared": len(py_rows), "mismatching_solutions": len(mismatches),
              "max_finish_time_absolute_difference": max_finish_difference,
              "cpp_results": str(cpp_directory.resolve()), "python_results": str(python_directory.resolve()),
              "cpp_run_created_utc": cpp_summary["created_utc"], "python_run_created_utc": py_summary["created_utc"],
              "timing_note": "Separate runs on recorded hardware; compare identical selected scenarios, seeds and budgets. Ratios describe these implementations, not intrinsic algorithm complexity.",
              "checks": ["same input identity and search settings", "identical task/robot routes", "same native completion and cost",
                         "same actual/best iteration counts", "finish-time agreement"],
              "timings": timings, "mismatches": mismatches}
    (python_directory / "backend_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    report = ["# Python / C++ 后端核对", "",
              "共核对 {} 个相同配置的解，状态：{}。".format(len(py_rows), result["status"]), "",
              "| 算法 | 场景数 | 完成率 | 平均 cost | Python 秒/场景 | C++ 秒/场景 | Python / C++ 时间比 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for algorithm, timing in timings.items():
        report.append("| {} | {} | {:.3f}% | {:.6f} | {:.6f} | {:.6f} | {:.2f} |".format(
            algorithm.upper(), timing["scenarios"], timing["completion_percent"], timing["mean_cost"],
            timing["python_seconds_per_scenario"], timing["cpp_seconds_per_scenario"], timing["python_over_cpp_time_ratio"]))
    report.extend(["", "比较相同场景、种子、参数、实际迭代数和补全规则。C++ 时间取自此前保存的对应实验，Python 为本次运行。",
                   "两次运行独立计时；倍率描述当前实现的实测差异，不代表算法本身的复杂度差异。",
                   "原生搜索和完整求解的计时规则没有因切换后端而改变。", ""])
    (python_directory / "BACKEND_COMPARISON.md").write_text("\n".join(report))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpp_directory", type=Path)
    parser.add_argument("python_directory", type=Path)
    options = parser.parse_args()
    result = compare(options.cpp_directory, options.python_directory)
    raise SystemExit(0 if result["status"] == "passed" else 1)
