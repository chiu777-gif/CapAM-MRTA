"""Audit persisted routes, metrics, grouping and dataset identity without solving."""
import argparse
from collections import defaultdict
import csv
from itertools import zip_longest
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from EILS.evaluate import summarize, taptc_group
from EILS.data_identity import dataset_identity
from EILS.navigation import load_evaluation_data, summarize_planning
from EILS.solver import prepare, sha256, validate_solution


def verify(output):
    output = Path(output)
    summary = json.loads((output / "summary.json").read_text())
    if summary.get("objective", "completion") not in ("completion", "cost"):
        raise ValueError("Unknown optimization objective")
    if dataset_identity(summary["dataset"])["sha256"] != summary["dataset_sha256"]:
        raise ValueError("Dataset hash changed")
    dataset, obstacle_map, _ = load_evaluation_data(
        summary["dataset"], summary["offset"], summary["scenarios_per_run"])
    cached = summary.get("path_planner") == "visibility_dijkstra"
    if obstacle_map is not None and not cached:
        raise ValueError("Obstacle result is missing visibility_dijkstra navigation metadata")
    if cached:
        snapshot = output / summary["path_distances_file"]
        if sha256(snapshot) != summary["path_distances_sha256"]:
            raise ValueError("Saved path distance snapshot hash changed")
        with np.load(snapshot, allow_pickle=False) as tables:
            prepared = [prepare(instance, {name: tables["{}_{}".format(name, index)]
                                          for name in ("task_to_task", "robot_to_task")})
                        for index, instance in enumerate(dataset)]
        with (output / "path_planning.csv").open() as handle:
            planning_rows = list(csv.DictReader(handle))
        if [int(row["scenario_index"]) for row in planning_rows] != list(
                range(summary["offset"], summary["offset"] + len(dataset))):
            raise ValueError("Path planning scenarios disagree with selected input")
        if [row["navigation_key"] for row in planning_rows] != summary["navigation_keys"]:
            raise ValueError("Path planning cache keys disagree")
        for row in planning_rows:
            if row["cache_hit"] not in ("True", "False"):
                raise ValueError("Invalid cache hit flag")
            row["cache_hit"] = row["cache_hit"] == "True"
            for name in ("path_planning_seconds", "path_planning_cpu_seconds",
                         "cache_read_seconds", "cache_write_seconds"):
                row[name] = float(row[name])
                if not np.isfinite(row[name]) or row[name] < 0:
                    raise ValueError("Invalid path timing")
            if row["cache_hit"] and row["path_planning_seconds"] != 0:
                raise ValueError("Cache hit must have zero path planning time")
        for name, value in summarize_planning(planning_rows).items():
            if not np.isclose(value, summary[name], atol=1e-10, rtol=1e-12):
                raise ValueError("Path planning summary mismatch: {}".format(name))
    else:
        prepared = [prepare(instance) for instance in dataset]
    rows = []
    keys = set()
    with (output / "scenarios.csv").open() as scenarios, (output / "routes.jsonl").open() as routes:
        for raw_row, route_line in zip_longest(csv.DictReader(scenarios), routes):
            if raw_row is None or route_line is None:
                raise ValueError("Route and scenario counts disagree")
            row = dict(raw_row)
            if row.get("objective", "completion") != summary.get("objective", "completion"):
                raise ValueError("Saved scenario objective disagrees with experiment configuration")
            if row.get("backend", "cpp") != summary.get("backend", "cpp"):
                raise ValueError("Saved backend disagrees with experiment configuration")
            for field in ("repeat", "scenario_index", "task_count", "n_agents", "seed", "tasks_done",
                          "native_completed_tasks", "iterations", "best_iteration"):
                row[field] = int(row[field])
            for field in ("cost", "completion_percent", "inference_seconds", "cpu_seconds"):
                row[field] = float(row[field])
            record = json.loads(route_line)
            if record.get("objective", "completion") != summary.get("objective", "completion"):
                raise ValueError("Saved route objective disagrees with experiment configuration")
            key = (row["algorithm"], row["profile"], row["repeat"], row["scenario_index"])
            if key in keys:
                raise ValueError("Duplicate result key: {}".format(key))
            keys.add(key)
            if key != (record["algorithm"], record["profile"], record["repeat"], record["scenario_index"]):
                raise ValueError("Route/scenario identifiers disagree")
            data = prepared[row["scenario_index"] - summary["offset"]]
            if "path_planner" in summary:
                index = row["scenario_index"] - summary["offset"]
                for saved in (row, record):
                    if (saved.get("path_planner") != summary["path_planner"]
                            or saved.get("navigation_key") != summary["navigation_keys"][index]):
                        raise ValueError("Saved route/scenario navigation metadata disagrees")
            input_name = dataset.names[row["scenario_index"] - summary["offset"]]
            original_group = {"group1": "tight", "group2": "slack"}.get(Path(input_name).parent.name)
            if original_group and row.get("deadline_group") != original_group:
                raise ValueError("Saved deadline group disagrees with original input directory")
            if row["n_agents"] != data["m"] or row["task_count"] != data["n"]:
                raise ValueError("Scenario dimensions disagree with input")
            expected_seed = (summary["seed"] + row["scenario_index"] + row["repeat"] * 1000003) % 2**64
            if row["seed"] != expected_seed:
                raise ValueError("Scenario seed mismatch")
            robots = np.full(data["n"], -1, dtype=np.int32)
            for robot, tour in enumerate(record["routes"]):
                for task in tour:
                    if not 0 <= task < data["n"] or robots[task] != -1:
                        raise ValueError("Invalid or duplicate task")
                    robots[task] = robot
            result = dict(routes=record["routes"], finish_times=np.array(record["finish_times"]),
                          robots=robots, native_completed_tasks=row["native_completed_tasks"])
            metrics = validate_solution(data, result, summary["service_time"])
            if metrics["tasks_done"] != row["tasks_done"] or not np.isclose(metrics["cost"], row["cost"], atol=1e-10, rtol=1e-12):
                raise ValueError("Saved cost/completion differs from independent replay")
            if not np.isclose(100 * metrics["tasks_done"] / data["n"], row["completion_percent"]):
                raise ValueError("Completion percentage mismatch")
            if not 1 <= row["best_iteration"] <= row["iterations"] or row["inference_seconds"] <= 0:
                raise ValueError("Invalid iteration/timing statistics")
            rows.append(row)
    profile_mode = summary.get("profile_mode", "fixed")
    expected = (len(prepared) * summary["repeats"] * len(summary["algorithms"])
                * (1 if profile_mode == "taptc_matched" else len(summary["profiles"])))
    expected_keys = {(algorithm, profile, repeat, index + summary["offset"])
                     for algorithm in summary["algorithms"] for profile in summary["profiles"]
                     for repeat in range(summary["repeats"]) for index in range(len(prepared))
                     if profile_mode != "taptc_matched" or profile == taptc_group(dataset.names[index])}
    if len(rows) != expected or keys != expected_keys:
        raise ValueError("Missing/extra scenario results")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["algorithm"], row["profile"], row["n_agents"])].append(row)
    for (algorithm, profile, m), items in grouped.items():
        group = summary["models"][algorithm + "_" + profile]["by_robot_count"][str(m)]
        for field, value in summarize(items).items():
            if not np.isclose(value, group[field], atol=1e-10, rtol=1e-12):
                raise ValueError("Summary metric mismatch: {}".format(field))
    for name, model in summary["models"].items():
        items = [row for row in rows if row["algorithm"] + "_" + row["profile"] == name]
        for field, value in summarize(items).items():
            if not np.isclose(value, model[field], atol=1e-10, rtol=1e-12):
                raise ValueError("Overall metric mismatch: {}".format(field))
    for algorithm, model in summary.get("matched_models", {}).items():
        items = [row for row in rows if row["algorithm"] == algorithm and row["profile"] == row.get("deadline_group")]
        if len(items) != len(prepared) * summary["repeats"]:
            raise ValueError("Missing group-matched scenarios")
        for field, value in summarize(items).items():
            if not np.isclose(value, model[field], atol=1e-10, rtol=1e-12):
                raise ValueError("Matched overall metric mismatch: {}".format(field))
        for dimension, source in (("by_robot_count", "n_agents"), ("by_deadline_group", "deadline_group")):
            for key, group in model[dimension].items():
                subset = [row for row in items if str(row[source]) == key]
                for field, value in summarize(subset).items():
                    if not np.isclose(value, group[field], atol=1e-10, rtol=1e-12):
                        raise ValueError("Matched group metric mismatch: {}".format(field))
    for group in summary.get("by_deadline_group", []):
        subset = [row for row in rows if row["algorithm"] == group["algorithm"]
                  and row["profile"] == group["profile"] and row["n_agents"] == group["n_agents"]
                  and row.get("deadline_group") == group["deadline_group"]]
        for field, value in summarize(subset).items():
            if not np.isclose(value, group[field], atol=1e-10, rtol=1e-12):
                raise ValueError("Deadline-group metric mismatch: {}".format(field))
    result = {"status": "passed", "solutions_replayed": len(rows),
              "groups_verified": len(grouped), "dataset_sha256": summary["dataset_sha256"],
              "matched_models_verified": len(summary.get("matched_models", {})),
              "deadline_groups_verified": len(summary.get("by_deadline_group", [])),
              "checks": ["dataset identity", "all algorithm/profile/scenario/repeat combinations present exactly once",
                         "all tasks assigned once to active robots", "full route finish times", "native completion unchanged",
                         "saved cost and completion", "seed derivation", "iteration/timing validity", "group and overall summaries"]}
    if cached:
        result["checks"].extend(["saved path distance snapshot identity", "cached-distance route replay",
                                  "navigation cache keys", "path planning timing and summaries"])
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    verify(parser.parse_args().output_dir)
