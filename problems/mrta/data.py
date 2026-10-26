"""Separate random training scenarios from fixed evaluation input.

Only normalize_instance converts physical data to model units. Versioned files
already contain model units and are never normalized a second time.
"""
from dataclasses import asdict
from pathlib import Path
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from .config import DATA_VERSION, DEFAULT_CONFIG

FLOAT_FIELDS = ('loc', 'depot', 'deadline', 'workload', 'max_speed',
                'robots_start_location', 'robots_work_capacity')


def validate_instance(instance):
    missing = set(FLOAT_FIELDS + ('n_agents',)) - set(instance)
    if missing:
        raise ValueError('Missing scenario fields: {}'.format(sorted(missing)))
    data = {key: torch.as_tensor(instance[key], dtype=torch.float32).clone()
            for key in FLOAT_FIELDS}
    count = torch.as_tensor(instance['n_agents']).item()
    if not np.isfinite(count) or count != int(count):
        raise ValueError('n_agents must be an integer')
    n_agents = int(count)
    if data['loc'].ndim != 2 or data['robots_start_location'].ndim != 2:
        raise ValueError('Task and robot coordinates must be matrices with two columns')
    n_tasks = data['loc'].shape[0]
    slots = data['robots_start_location'].shape[0]
    shapes = {'loc': (n_tasks, 2), 'depot': (1, 2), 'deadline': (n_tasks,),
              'workload': (n_tasks,), 'robots_start_location': (slots, 2),
              'robots_work_capacity': (slots,)}
    for name, shape in shapes.items():
        if tuple(data[name].shape) != shape:
            raise ValueError('{} must have shape {}, got {}'.format(name, shape, tuple(data[name].shape)))
    if n_tasks < 1 or not 1 <= n_agents <= slots:
        raise ValueError('A scenario needs tasks and 1 <= n_agents <= robot slots')
    if data['max_speed'].numel() != 1:
        raise ValueError('Each scenario must have one scalar max_speed')
    data['max_speed'] = data['max_speed'].reshape(())
    for name, value in data.items():
        if not torch.isfinite(value).all():
            raise ValueError('{} contains non-finite values'.format(name))
    for name in ('deadline', 'max_speed', 'robots_work_capacity'):
        if (data[name] <= 0).any():
            raise ValueError('{} must be positive'.format(name))
    if (data['workload'] < 0).any():
        raise ValueError('workload must be nonnegative')
    data['n_agents'] = torch.tensor(n_agents, dtype=torch.long)
    data['max_n_agents'] = torch.tensor(slots, dtype=torch.long)
    return data


def normalize_instance(raw, config):
    """Physical map/work units -> model units. All times remain in seconds."""
    return validate_instance({
        'loc': torch.as_tensor(raw['loc']) / config.coordinate_scale,
        'depot': torch.as_tensor(raw.get('depot', [[0.0, 0.0]])) / config.coordinate_scale,
        'deadline': raw['deadline'],
        'workload': torch.as_tensor(raw['workload']) / config.workload_scale,
        'n_agents': raw['n_agents'],
        'max_speed': torch.as_tensor(raw.get('max_speed', config.agent_max_speed)) / config.coordinate_scale,
        'robots_start_location': torch.as_tensor(raw['robots_start_location']) / config.coordinate_scale,
        'robots_work_capacity': torch.as_tensor(raw['robots_work_capacity']) / config.workload_scale,
    })


class RandomMRTADataset(Dataset):
    """Lazy, repeatable uniform scenarios; contains no TAPTC grouping rules."""
    def __init__(self, config, num_samples, seed):
        if num_samples < 1:
            raise ValueError('num_samples must be positive')
        self.config, self.size, self.seed = config, num_samples, seed

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if not 0 <= index < self.size:
            raise IndexError(index)
        config = self.config
        generator = torch.Generator().manual_seed((self.seed + index) % (2 ** 63))

        def uniform(shape, low, high):
            return torch.empty(shape).uniform_(low, high, generator=generator)

        slots = max(config.robot_counts)
        robot_index = torch.randint(len(config.robot_counts), (), generator=generator).item()
        return normalize_instance({
            'loc': uniform((config.graph_size, 2), 0, config.map_size),
            'deadline': uniform((config.graph_size,), config.deadline_min, config.deadline_max),
            'workload': uniform((config.graph_size,), config.workload_min, config.workload_max),
            'n_agents': config.robot_counts[robot_index],
            'robots_start_location': uniform((slots, 2), 0, config.map_size),
            'robots_work_capacity': uniform((slots,), config.work_capacity_min, config.work_capacity_max),
        }, config)


class FileMRTADataset(Dataset):
    def __init__(self, instances, names=None):
        if not instances:
            raise ValueError('Dataset is empty')
        self.data = instances
        self.names = names if names is not None else [str(i) for i in range(len(instances))]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def collate_mrta(instances):
    """Pad robot slots, not tasks. Padded robots can never take decisions."""
    if len({item['loc'].size(0) for item in instances}) != 1:
        raise ValueError('A batch must have equal task counts; use --eval_batch_size 1 for mixed graphs')
    slots = max(item['robots_start_location'].size(0) for item in instances)
    result = {}
    for key in instances[0]:
        values = []
        for item in instances:
            value = item[key]
            if key in ('robots_start_location', 'robots_work_capacity'):
                pad = slots - value.size(0)
                padding = value.new_zeros((pad, 2)) if value.ndim == 2 else value.new_ones(pad)
                value = torch.cat((value, padding))
            elif key == 'max_n_agents':
                value = value.new_tensor(slots)
            values.append(value)
        result[key] = torch.stack(values)
    return result


def _dictionary_instance(data, config, data_format):
    loc, robots = data['loc_data'], data['robot_data']
    raw = {'loc': loc['loc'], 'workload': loc['workload'], 'deadline': loc['deadline'],
           'robots_start_location': robots['robots_loc'],
           'robots_work_capacity': robots['robots_capacity'],
           'n_agents': len(robots['robots_capacity'])}
    if data_format == 'taptc':
        return normalize_instance(raw, config)
    if data_format != 'legacy_normalized':
        raise ValueError('Untagged dictionary has ambiguous units. Specify --data_format '
                         'legacy_normalized (old converted files) or taptc (physical units).')
    raw['depot'] = [[0.0, 0.0]]
    raw['max_speed'] = config.agent_max_speed / config.coordinate_scale
    return validate_instance(raw)


def _read_taptc(path, config):
    # r12a3i00.txt -> agent/a3i00.txt, in the original TAPTC layout.
    try:
        agent_name = 'a' + path.stem.split('a', 1)[1] + '.txt'
    except IndexError:
        raise ValueError('Expected a TAPTC task filename such as r12a3i00.txt: {}'.format(path))
    robot_path = path.parent.parent / 'agent' / agent_name
    tasks = np.loadtxt(str(path), skiprows=1, ndmin=2)
    robots = np.loadtxt(str(robot_path), skiprows=1, ndmin=2)
    if tasks.shape[1] != 5 or robots.shape[1] != 4:
        raise ValueError('Invalid TAPTC task/robot columns: {}'.format(path))
    return normalize_instance({
        'loc': tasks[:, 1:3], 'workload': tasks[:, 3], 'deadline': tasks[:, 4],
        'n_agents': len(robots), 'robots_start_location': robots[:, 1:3],
        'robots_work_capacity': robots[:, 3],
    }, config)


def load_file_dataset(paths, config=DEFAULT_CONFIG, data_format='auto', offset=0, num_samples=None):
    """Read fixed scenarios; never generate, replace deadlines, or crop tasks."""
    if offset < 0 or (num_samples is not None and num_samples < 1):
        raise ValueError('offset must be nonnegative and num_samples positive')
    paths = [paths] if isinstance(paths, (str, Path)) else paths
    instances, names, active = [], [], set()

    def read(path):
        path = Path(path)
        if path.is_dir():
            files = sorted(path.glob('group[12]/r*.txt'))
            if not files:
                raise ValueError('No TAPTC group1/group2 task files in {}'.format(path))
            for item in files:
                read(item)
            return
        resolved = path.resolve()
        if resolved in active:
            raise ValueError('Cyclic dataset index: {}'.format(path))
        if not path.is_file():
            raise FileNotFoundError('Dataset does not exist: {}'.format(path))
        active.add(resolved)
        try:
            if path.suffix == '.txt':
                instances.append(_read_taptc(path, config))
                names.append(str(path))
                return
            if path.suffix != '.pkl':
                raise ValueError('Expected .pkl data or TAPTC .txt: {}'.format(path))
            with path.open('rb') as handle:
                data = pickle.load(handle)
            if isinstance(data, dict) and data.get('format') == 'capam_mrta':
                if data.get('version') != DATA_VERSION or data.get('units', {}).get('time') != 'seconds':
                    raise ValueError('Unsupported dataset version/units: {}'.format(path))
                if data['units'].get('representation') != 'normalized':
                    raise ValueError('Versioned datasets must already use normalized model units')
                for name in ('coordinate_scale', 'workload_scale'):
                    if data['units'].get(name) != getattr(config, name):
                        raise ValueError('Dataset {}={} differs from model configuration {}. '
                                         'Use matching unit scales; normalized tensors are never '
                                         'silently rescaled.'.format(name, data['units'].get(name),
                                                                    getattr(config, name)))
                for index, item in enumerate(data['instances']):
                    instances.append(validate_instance(item))
                    names.append('{}:{}'.format(path, index))
            elif isinstance(data, dict) and 'loc_data' in data and 'robot_data' in data:
                instances.append(_dictionary_instance(data, config, data_format))
                names.append(str(path))
            elif isinstance(data, list) and data and all(isinstance(item, str) for item in data):
                for item in data:
                    candidates = (Path(item), path.parent / item, path.parent / Path(item).name)
                    target = next((candidate for candidate in candidates if candidate.is_file()), None)
                    if target is None:
                        raise FileNotFoundError('Index {} references missing {}. For original TAPTC '
                                                'data, pass its directory directly.'.format(path, item))
                    read(target)
            else:
                raise ValueError('Unsupported old dataset (e.g. depot/loc/deadline tuples lack robot '
                                 'and workload fields): {}'.format(path))
        finally:
            active.remove(resolved)

    for path in paths:
        read(path)
    end = None if num_samples is None else offset + num_samples
    return FileMRTADataset(instances[offset:end], names[offset:end])


def save_training_dataset(dataset, path, overwrite=False):
    path = Path(path)
    if path.suffix != '.pkl':
        path = Path(str(path) + '.pkl')
    if path.exists() and not overwrite:
        raise FileExistsError('Use -f to overwrite {}'.format(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {'format': 'capam_mrta', 'version': DATA_VERSION,
               'units': {'representation': 'normalized', 'time': 'seconds',
                         'coordinate_scale': dataset.config.coordinate_scale,
                         'workload_scale': dataset.config.workload_scale},
               'generation': asdict(dataset.config), 'seed': dataset.seed,
               'instances': [dataset[index] for index in range(len(dataset))]}
    with path.open('wb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path
