"""Obstacle input and visibility caches shared with z_test_path."""
from pathlib import Path
import pickle
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import multiprocessing
import os

from problems.mrta.config import DEFAULT_CONFIG
from problems.mrta.data import FileMRTADataset, load_file_dataset
from z_test_path.path_planner import load_obstacle_data, precompute_navigation


def load_evaluation_data(path, offset=0, num_samples=None):
    path = Path(path).resolve()
    if path.is_file() and path.suffix == '.pkl':
        with path.open('rb') as handle:
            payload = pickle.load(handle)
        if isinstance(payload, dict) and 'obstacle_map' in payload:
            config, obstacle_map, instances = load_obstacle_data(path)
            end = None if num_samples is None else offset + num_samples
            indices = list(range(len(instances)))[offset:end]
            data = FileMRTADataset([instances[i] for i in indices],
                                   ['{}:{}'.format(path, i) for i in indices])
            return data, obstacle_map, config
    return load_file_dataset(path, offset=offset, num_samples=num_samples), None, DEFAULT_CONFIG


def plan(instance, obstacle_map, config, cache_dir, epsilon, rebuild=False):
    navigation, stats = precompute_navigation(
        instance, obstacle_map, config.map_size / config.coordinate_scale,
        cache_dir=cache_dir, rebuild=rebuild, path_planner='visibility_dijkstra',
        visibility_epsilon=epsilon)
    # Keep only distance tables in memory; geometry remains in the disk cache.
    tables = {key: navigation[key] for key in ('task_to_task', 'robot_to_task')}
    return tables, dict(navigation_key=navigation['key'], cache_hit=stats['cache_hit'],
                        path_planning_seconds=stats['planning_seconds'],
                        path_planning_cpu_seconds=stats['planning_cpu_seconds'],
                        cache_read_seconds=stats['cache_read_seconds'],
                        cache_write_seconds=stats['cache_write_seconds'])


@contextmanager
def path_worker_environment():
    """Bound numerical threads before spawned workers import NumPy/PyTorch."""
    names = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
    previous = {name: os.environ.get(name) for name in names}
    try:
        os.environ.update({name: '1' for name in names})
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def initialize_path_worker():
    import torch
    torch.set_num_threads(1)


def plan_job(job):
    import torch
    index, arrays, obstacle_map, config, cache_dir, epsilon, rebuild = job
    instance = {name: torch.as_tensor(value) for name, value in arrays.items()}
    tables, stats = plan(instance, obstacle_map, config, cache_dir, epsilon, rebuild)
    return index, tables, stats


def plan_scenarios(instances, obstacle_map, config, cache_dir, epsilon, rebuild=False, workers=1):
    """Yield indexed NumPy distance tables; geometry stays in the shared cache."""
    if workers < 1:
        raise ValueError('path_workers must be positive')
    workers = min(workers, len(instances))
    jobs = ((index, {name: instance[name].detach().cpu().numpy()
                     for name in ('loc', 'robots_start_location', 'n_agents')},
             obstacle_map, config, cache_dir, epsilon, rebuild)
            for index, instance in enumerate(instances))
    if workers <= 1:
        yield from map(plan_job, jobs)
    else:
        # NumPy crosses process boundaries, avoiding PyTorch shared-memory IPC.
        with path_worker_environment(), ProcessPoolExecutor(
                max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize_path_worker) as pool:
            yield from pool.map(plan_job, jobs, chunksize=1)
