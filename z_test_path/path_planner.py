"""Precompute obstacle-aware distances with selectable navigation algorithms."""
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import tempfile
import time

import numpy as np
import torch

from problems.mrta.config import DATA_VERSION, MRTAConfig
from problems.mrta.data import FileMRTADataset, validate_instance
from z_test_path import cankao_planner, visibility_planner
from z_test_path.generate_test_set import ObstacleMap
from z_test_path.visibility_planner import DEFAULT_VISIBILITY_EPSILON, VisibilityGraphPlanner

CACHE_VERSION = 2
PLANNER_DESCRIPTIONS = {
    'grid_astar': 'cankao_planner.GridPathPlanner: 8-neighbor A* + collision-checked smoothing',
    'visibility_dijkstra': 'visibility_planner.VisibilityGraphPlanner: visibility graph + Dijkstra',
}


def planner_settings(path_planner='grid_astar', subdivisions=10,
                     visibility_epsilon=DEFAULT_VISIBILITY_EPSILON):
    if path_planner not in PLANNER_DESCRIPTIONS:
        raise ValueError('Unknown path_planner: {}'.format(path_planner))
    return {
        'path_planner': path_planner, 'planner': PLANNER_DESCRIPTIONS[path_planner],
        'path_subdivisions': subdivisions if path_planner == 'grid_astar' else None,
        'visibility_epsilon': visibility_epsilon if path_planner == 'visibility_dijkstra' else None,
    }


def load_obstacle_data(path, model_config=None):
    with Path(path).open('rb') as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or payload.get('format') != 'capam_mrta':
        raise ValueError('Expected a versioned obstacle dataset from generate_test_set.py')
    units = payload.get('units', {})
    if (payload.get('version') != DATA_VERSION or units.get('representation') != 'normalized'
            or units.get('time') != 'seconds'):
        raise ValueError('Unsupported dataset version or units')
    config = MRTAConfig(**payload['generation'])
    for name in ('coordinate_scale', 'workload_scale'):
        if units.get(name) != getattr(config, name):
            raise ValueError('Dataset {} differs from its generation parameters'.format(name))
        if model_config is not None and units[name] != getattr(model_config, name):
            raise ValueError('Dataset {}={} differs from model {}'.format(
                name, units[name], getattr(model_config, name)))
    obstacle_map = ObstacleMap(payload['obstacle_map']['grid'])
    instances = [validate_instance(item) for item in payload['instances']]
    if not instances:
        raise ValueError('Dataset is empty')
    return config, obstacle_map, instances


def make_planner(obstacle_map, subdivisions=10, path_planner='grid_astar',
                 visibility_epsilon=DEFAULT_VISIBILITY_EPSILON):
    planner_settings(path_planner, subdivisions, visibility_epsilon)
    height, width = obstacle_map.height, obstacle_map.width
    rectangles = [(c / width, (height - r - 1) / height,
                   (c + 1) / width, (height - r) / height)
                  for r, row in enumerate(obstacle_map.grid)
                  for c, symbol in enumerate(row) if symbol == '#']
    if path_planner == 'visibility_dijkstra':
        return VisibilityGraphPlanner(rows=obstacle_map.grid, obstacles=rectangles, epsilon=visibility_epsilon)
    return cankao_planner.GridPathPlanner(rows=obstacle_map.grid, obstacles=rectangles, subdivisions=subdivisions)


def navigation_key(instance, obstacle_map, extent, subdivisions, path_planner='grid_astar',
                   visibility_epsilon=DEFAULT_VISIBILITY_EPSILON):
    digest = hashlib.sha256()
    settings = planner_settings(path_planner, subdivisions, visibility_epsilon)
    modules = [cankao_planner] + ([visibility_planner] if path_planner == 'visibility_dijkstra' else [])
    planner_class = (VisibilityGraphPlanner if path_planner == 'visibility_dijkstra'
                     else cankao_planner.GridPathPlanner)
    specification = {
        'version': CACHE_VERSION, 'planner_version': planner_class.VERSION,
        'planner_source_sha256': {module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                                  for module in modules},
        'grid': obstacle_map.grid, 'extent': extent, **settings,
        'n_agents': int(instance['n_agents']),
    }
    digest.update(json.dumps(specification, sort_keys=True).encode('utf-8'))
    for name in ('loc', 'robots_start_location'):
        array = instance[name].detach().cpu().numpy().astype('<f8')
        digest.update(str(array.shape).encode('ascii'))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _validate_navigation(navigation, key, tasks, slots, agents):
    if navigation.get('version') != CACHE_VERSION or navigation.get('key') != key:
        raise ValueError('Navigation cache has an incompatible version or fingerprint')
    task_distances = navigation['task_to_task']
    start_distances = navigation['robot_to_task']
    if task_distances.shape != (tasks, tasks) or start_distances.shape != (slots, tasks):
        raise ValueError('Navigation cache matrix shapes do not match the scenario')
    for values in (task_distances, start_distances):
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('Navigation cache contains invalid distances')
    if not np.array_equal(task_distances, task_distances.T) or np.any(np.diag(task_distances)):
        raise ValueError('Task distance cache must be symmetric with a zero diagonal')
    if len(navigation['paths']) != tasks * (tasks - 1) // 2 + agents * tasks:
        raise ValueError('Navigation cache has missing paths; use --rebuild_cache')


def precompute_navigation(instance, obstacle_map, extent, subdivisions=10,
                          cache_dir=None, rebuild=False, path_planner='grid_astar',
                          visibility_epsilon=DEFAULT_VISIBILITY_EPSILON):
    """Distances use model units; both planners operate on [0,1]^2.

    Node IDs here are zero-based: tasks first, then active robot starts. The
    depot and inactive robot slots do not participate in navigation.
    """
    if not math.isfinite(extent) or extent <= 0:
        raise ValueError('Normalized map extent must be finite and positive')
    planner = make_planner(obstacle_map, subdivisions, path_planner, visibility_epsilon)
    tasks, slots = len(instance['loc']), len(instance['robots_start_location'])
    agents = int(instance['n_agents'])
    key = navigation_key(instance, obstacle_map, extent, subdivisions, path_planner, visibility_epsilon)
    cache_path = Path(cache_dir) / (key + '.pkl') if cache_dir is not None else None
    started = time.perf_counter()
    cpu_started = time.process_time()
    if cache_path is not None and cache_path.exists() and not rebuild:
        with cache_path.open('rb') as handle:
            navigation = pickle.load(handle)
        _validate_navigation(navigation, key, tasks, slots, agents)
        return navigation, {'cache_hit': True, 'planning_seconds': 0.,
                            'planning_cpu_seconds': 0.,
                            'cache_read_seconds': time.perf_counter() - started,
                            'cache_write_seconds': 0.}

    positions = np.concatenate((instance['loc'].cpu().numpy(),
                                instance['robots_start_location'][:agents].cpu().numpy())).astype(float)
    unit_positions = positions / extent
    for node, point in enumerate(unit_positions):
        try:
            planner.validate_point(point)
        except ValueError as error:
            raise ValueError('Navigation node {} is invalid: {}'.format(node, error)) from error
    task_to_task = np.zeros((tasks, tasks), dtype=np.float32)
    robot_to_task = np.zeros((slots, tasks), dtype=np.float32)
    paths = {}
    for first in range(tasks):
        # No robot-to-robot distances are needed by the event-driven simulator.
        for second in range(first + 1, tasks + agents):
            try:
                path, distance = planner.plan(unit_positions[first], unit_positions[second])
            except ValueError as error:
                raise ValueError('Nodes {} -> {} are unreachable: {}'.format(first, second, error)) from error
            paths[first, second] = path * extent
            if second < tasks:
                task_to_task[first, second] = task_to_task[second, first] = distance * extent
            else:
                robot_to_task[second - tasks, first] = distance * extent
    navigation = {
        'version': CACHE_VERSION, 'key': key, 'extent': extent,
        **planner_settings(path_planner, subdivisions, visibility_epsilon),
        'subdivisions': subdivisions if path_planner == 'grid_astar' else None,
        'positions': positions, 'paths': paths,
        'task_to_task': task_to_task, 'robot_to_task': robot_to_task,
    }
    _validate_navigation(navigation, key, tasks, slots, agents)
    stats = {'cache_hit': False, 'planning_seconds': time.perf_counter() - started,
             'planning_cpu_seconds': time.process_time() - cpu_started,
             'cache_read_seconds': 0., 'cache_write_seconds': 0.}
    if cache_path is not None:
        started = time.perf_counter()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=str(cache_path.parent), delete=False) as handle:
                temporary = Path(handle.name)
                pickle.dump(navigation, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(str(temporary), str(cache_path))
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        stats['cache_write_seconds'] = time.perf_counter() - started
    return navigation, stats


def get_path(navigation, start, goal):
    if start == goal:
        return np.repeat(navigation['positions'][start:start + 1], 2, axis=0)
    path = navigation['paths'][min(start, goal), max(start, goal)]
    return path if start < goal else path[::-1]


def prepare_path_dataset(instances, indices, navigations, source):
    """Pad all slots before the common collator stacks the extra table.

    Table rows: depot placeholder, tasks, robot starts. Columns: depot
    placeholder, tasks. Placeholder entries are never selected as actions.
    """
    slots = max(len(item['robots_start_location']) for item in instances)
    prepared = []
    for item, navigation in zip(instances, navigations):
        sample = dict(item)
        tasks, old_slots = len(item['loc']), len(item['robots_start_location'])
        if old_slots < slots:
            sample['robots_start_location'] = torch.cat((item['robots_start_location'],
                                                         torch.zeros(slots - old_slots, 2)))
            sample['robots_work_capacity'] = torch.cat((item['robots_work_capacity'],
                                                        torch.ones(slots - old_slots)))
        sample['max_n_agents'] = torch.tensor(slots, dtype=torch.long)
        table = torch.zeros(1 + tasks + slots, 1 + tasks)
        table[1:tasks + 1, 1:] = torch.from_numpy(navigation['task_to_task'])
        table[tasks + 1:tasks + 1 + old_slots, 1:] = torch.from_numpy(navigation['robot_to_task'])
        sample['path_distances'] = table
        prepared.append(sample)
    dataset = FileMRTADataset(prepared, ['{}:{}'.format(source, index) for index in indices])
    dataset.indices = list(indices)
    return dataset
