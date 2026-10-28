"""Reuse z_test_path visibility graphs and model-unit navigation caches."""
from pathlib import Path
import pickle

import torch

from problems.mrta.data import FileMRTADataset, load_file_dataset
from z_test_path.path_planner import load_obstacle_data, precompute_navigation


def load_evaluation_data(path, offset=0, num_samples=None):
    path = Path(path).resolve()
    if path.suffix == ".pkl" and path.is_file():
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if isinstance(payload, dict) and "obstacle_map" in payload:
            config, obstacle_map, instances = load_obstacle_data(path)
            end = None if num_samples is None else offset + num_samples
            indices = list(range(len(instances)))[offset:end]
            dataset = FileMRTADataset([instances[index] for index in indices],
                                      ["{}:{}".format(path, index) for index in indices])
            return dataset, obstacle_map, config.map_size / config.coordinate_scale
    return load_file_dataset([path], offset=offset, num_samples=num_samples), None, None


def plan_scenario(job):
    index, instance, obstacle_map, extent, cache_dir, epsilon, rebuild = job
    # NumPy crosses process boundaries without PyTorch shared-memory sockets.
    instance = {name: torch.as_tensor(value) for name, value in instance.items()}
    navigation, stats = precompute_navigation(
        instance, obstacle_map, extent, cache_dir=cache_dir, rebuild=rebuild,
        path_planner="visibility_dijkstra", visibility_epsilon=epsilon)
    # Geometry remains in the shared cache; search workers only need these tables.
    tables = {name: navigation[name] for name in ("task_to_task", "robot_to_task")}
    row = dict(scenario_index=index, navigation_key=navigation["key"],
               cache_hit=stats["cache_hit"], path_planning_seconds=stats["planning_seconds"],
               path_planning_cpu_seconds=stats["planning_cpu_seconds"],
               cache_read_seconds=stats["cache_read_seconds"],
               cache_write_seconds=stats["cache_write_seconds"])
    return index, tables, row


def summarize_planning(rows):
    return dict(path_scenarios=len(rows), cache_hits=sum(row["cache_hit"] for row in rows),
                cache_misses=sum(not row["cache_hit"] for row in rows),
                path_planning_seconds_total=sum(row["path_planning_seconds"] for row in rows),
                path_planning_cpu_seconds_total=sum(row["path_planning_cpu_seconds"] for row in rows),
                cache_read_seconds_total=sum(row["cache_read_seconds"] for row in rows),
                cache_write_seconds_total=sum(row["cache_write_seconds"] for row in rows))
