"""Generate MRTA test scenarios whose coordinates avoid an ASCII obstacle map."""
import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import pickle
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
from torch.utils.data import Dataset

from problems.mrta.config import (
    DATA_VERSION, DEFAULT_CONFIG, DEFAULT_SEED, MRTAConfig,
    add_environment_arguments, split_seed,
)
from problems.mrta.data import RandomMRTADataset


@dataclass(frozen=True)
class ObstacleMap:
    """Equal-area cells covering a square, with text row 0 at the top."""

    grid: tuple

    def __post_init__(self):
        object.__setattr__(self, 'grid', tuple(self.grid))
        if not self.grid or not self.grid[0]:
            raise ValueError('Obstacle map must not be empty')
        width = len(self.grid[0])
        for number, row in enumerate(self.grid, 1):
            if len(row) != width:
                raise ValueError('Map row {} has width {}; expected {}'.format(number, len(row), width))
            invalid = set(row) - {'.', '#'}
            if invalid:
                raise ValueError('Map row {} contains invalid symbols {}; only . and # are allowed'.format(
                    number, repr(''.join(sorted(invalid)))))
        if not any('.' in row for row in self.grid):
            raise ValueError('Obstacle map must contain at least one free cell (.)')

    @classmethod
    def from_file(cls, path):
        return cls(tuple(Path(path).read_text(encoding='utf-8-sig').splitlines()))

    @property
    def height(self):
        return len(self.grid)

    @property
    def width(self):
        return len(self.grid[0])

    @property
    def free_cells(self):
        return [(row, column) for row, line in enumerate(self.grid)
                for column, symbol in enumerate(line) if symbol == '.']

    def metadata(self, config):
        free_count = len(self.free_cells)
        return {
            'grid': list(self.grid), 'rows': self.height, 'columns': self.width,
            'free_cells': free_count, 'obstacle_cells': self.height * self.width - free_count,
            'symbols': {'#': 'obstacle', '.': 'free'},
            'origin': 'bottom-left; x increases right; y increases up; first text row is top',
            'physical_cell_size': [config.map_size / self.width, config.map_size / self.height],
            'normalized_cell_size': [config.map_size / config.coordinate_scale / self.width,
                                     config.map_size / config.coordinate_scale / self.height],
        }


class ObstacleMRTADataset(Dataset):
    """Uniform continuous positions in free cells, including a random depot."""

    def __init__(self, config, num_samples, seed, obstacle_map):
        self.base = RandomMRTADataset(config, num_samples, seed)
        self.config, self.seed, self.obstacle_map = config, seed, obstacle_map
        self.free_cells = torch.tensor(obstacle_map.free_cells, dtype=torch.long)

    def __len__(self):
        return len(self.base)

    def _sample_positions(self, count, generator):
        # Equal-area cells: uniform cell selection + uniform offsets gives a
        # uniform spatial distribution over the entire free region.
        chosen = self.free_cells[torch.randint(len(self.free_cells), (count,), generator=generator)]
        rows, columns = chosen[:, 0].double(), chosen[:, 1].double()
        width, height = self.obstacle_map.width, self.obstacle_map.height
        extent = self.config.map_size / self.config.coordinate_scale
        lower = torch.stack((columns / width, (height - rows - 1) / height), dim=-1) * extent
        upper = torch.stack(((columns + 1) / width, (height - rows) / height), dim=-1) * extent
        result = torch.empty(count, 2, dtype=torch.float32)
        pending = torch.arange(count)
        # Validate AFTER float32 conversion so rounding cannot put a stored
        # point onto a cell edge or inside an adjacent obstacle.
        for _ in range(64):
            offsets = torch.rand(len(pending), 2, dtype=torch.float64, generator=generator)
            candidate = (lower[pending] + offsets * (upper[pending] - lower[pending])).float()
            inside = ((candidate.double() > lower[pending]) &
                      (candidate.double() < upper[pending])).all(dim=-1)
            result[pending[inside]] = candidate[inside]
            pending = pending[~inside]
            if len(pending) == 0:
                return result
        raise ValueError('Map cells have insufficient float32 coordinate precision; '
                         'use a coarser grid or adjust map_size/coordinate_scale')

    def __getitem__(self, index):
        item = self.base[index]
        generator = torch.Generator().manual_seed(split_seed(self.seed, 'obstacle_positions', index))
        tasks, slots = self.config.graph_size, item['robots_start_location'].size(0)
        positions = self._sample_positions(tasks + slots + 1, generator)
        item['loc'] = positions[:tasks].clone()
        item['robots_start_location'] = positions[tasks:tasks + slots].clone()
        item['depot'] = positions[-1:].clone()
        return item


def save_dataset(dataset, path, source_map, overwrite=False):
    """Keep the standard scenario schema; persist the map at file level."""
    path = Path(path)
    metadata_path = path.with_suffix('.json')
    for target in (path, metadata_path):
        if target.exists() and not overwrite:
            raise FileExistsError('Already exists: {}. Use --overwrite to replace it.'.format(target))
    config = dataset.config
    units = {'representation': 'normalized', 'time': 'seconds',
             'coordinate_scale': config.coordinate_scale, 'workload_scale': config.workload_scale}
    obstacle_map = dataset.obstacle_map.metadata(config)
    distributions = {
        'task_coordinates': 'continuous uniform over all free cells',
        'robot_initial_coordinates': 'continuous uniform over all free cells, including inactive slots',
        'depot_coordinates': 'continuous uniform over all free cells; resampled per scenario',
        'robot_count': 'uniform choice from robot_counts; n_agents fixes one count',
        'deadline': 'continuous uniform [deadline_min, deadline_max]',
        'workload': 'continuous uniform [workload_min, workload_max]',
        'work_capacity': 'continuous uniform [work_capacity_min, work_capacity_max]',
    }
    payload = {
        'format': 'capam_mrta', 'version': DATA_VERSION, 'units': units,
        'generation': asdict(config), 'seed': dataset.seed,
        'obstacle_map': obstacle_map, 'distributions': distributions,
        'path_distances_precomputed': False,
        'instances': [dataset[index] for index in range(len(dataset))],
    }
    metadata = {
        'dataset': str(path.resolve()), 'scenarios': len(dataset),
        'tasks_per_scenario': config.graph_size, 'seed': dataset.seed,
        'physical_parameters': asdict(config), 'units': units,
        'source_map': str(Path(source_map).resolve()), 'obstacle_map': obstacle_map,
        'distributions': distributions, 'path_distances_precomputed': False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb' if overwrite else 'xb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with metadata_path.open('w' if overwrite else 'x', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2)
        handle.write('\n')
    return path


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_environment_arguments(parser, include_graph=False)
    for action in parser._actions:
        if action.dest == 'robot_counts':
            action.help = 'Robot counts sampled uniformly for each generated scenario'
        elif action.dest == 'n_agents':
            action.help = 'Use exactly this many robots per generated scenario'
    parser.add_argument('--map_file', '--obstacle_map', type=Path, required=True,
                        help='Rectangular ASCII map: # is blocked, . is free; first row is top')
    parser.add_argument('--graph_sizes', '--graph_size', type=int, nargs='+',
                        default=[DEFAULT_CONFIG.graph_size], help='Tasks per scenario')
    parser.add_argument('--dataset_size', type=int, default=100, help='Scenarios for EACH task size')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--output_dir', type=Path, default=Path(__file__).resolve().parent / 'data')
    parser.add_argument('--filename', '--output_file', type=Path,
                        help='Explicit .pkl filename; requires exactly one task size')
    parser.add_argument('--overwrite', '-f', action='store_true')
    opts = parser.parse_args(args)
    if opts.dataset_size < 1:
        parser.error('dataset_size must be positive')
    if not 0 <= opts.seed < 2 ** 63:
        parser.error('seed must be in [0, 2**63)')
    if len(set(opts.graph_sizes)) != len(opts.graph_sizes):
        parser.error('graph_sizes must be distinct')
    if opts.filename and (len(opts.graph_sizes) != 1 or opts.filename.suffix != '.pkl'):
        parser.error('filename must end in .pkl and requires exactly one task size')

    try:
        obstacle_map = ObstacleMap.from_file(opts.map_file)
        base_config = MRTAConfig.from_options(opts)
        planned = []
        for size in opts.graph_sizes:
            config = replace(base_config, graph_size=size)
            robots = '-'.join(str(n) for n in config.robot_counts)
            path = opts.filename or opts.output_dir / 'n{}_m{}_{}_seed{}.pkl'.format(
                size, robots, opts.map_file.stem, opts.seed)
            for target in (path, path.with_suffix('.json')):
                if target.exists() and not opts.overwrite:
                    raise FileExistsError('Already exists: {}. Use --overwrite to replace it.'.format(target))
            planned.append((config, path))

        outputs = []
        for config, path in planned:
            dataset = ObstacleMRTADataset(config, opts.dataset_size, opts.seed, obstacle_map)
            save_dataset(dataset, path, opts.map_file, overwrite=opts.overwrite)
            print('Saved {} scenarios, {} tasks/scenario, robot counts {}: {}'.format(
                len(dataset), config.graph_size, config.robot_counts, path))
            print('Map: {} rows x {} columns; {} free cells, {} obstacle cells'.format(
                obstacle_map.height, obstacle_map.width, len(dataset.free_cells),
                obstacle_map.height * obstacle_map.width - len(dataset.free_cells)))
            print('Parameters and obstacle map: {}'.format(path.with_suffix('.json')))
            outputs.append(path)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return outputs


if __name__ == '__main__':
    main()
