"""Generate fixed MRTA test datasets from physical parameters."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from problems.mrta.config import DEFAULT_CONFIG, DEFAULT_SEED, MRTAConfig, add_environment_arguments
from problems.mrta.data import RandomMRTADataset, save_training_dataset


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_environment_arguments(parser, include_graph=False)
    for action in parser._actions:
        if action.dest == 'robot_counts':
            action.help = 'Robot counts sampled uniformly for each generated scenario'
        elif action.dest == 'n_agents':
            action.help = 'Use exactly this many robots per generated scenario'
    parser.add_argument('--graph_sizes', '--graph_size', type=int, nargs='+',
                        default=[DEFAULT_CONFIG.graph_size], help='Tasks per scenario')
    parser.add_argument('--dataset_size', type=int, default=100,
                        help='Scenarios to generate for EACH task size')
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
        base = MRTAConfig.from_options(opts)
        planned = []
        for size in opts.graph_sizes:
            config = replace(base, graph_size=size)
            robots = '-'.join(str(n) for n in config.robot_counts)
            path = opts.filename or opts.output_dir / 'n{}_m{}_seed{}.pkl'.format(size, robots, opts.seed)
            if not opts.overwrite:
                for target in (path, path.with_suffix('.json')):
                    if target.exists():
                        raise FileExistsError('Already exists: {}. Choose another path or use --overwrite.'.format(target))
            planned.append((config, path))
        outputs = []
        for config, path in planned:
            dataset = RandomMRTADataset(config, opts.dataset_size, opts.seed)
            save_training_dataset(dataset, path, overwrite=opts.overwrite)
            metadata = {
                'dataset': str(path.resolve()), 'scenarios': len(dataset),
                'tasks_per_scenario': config.graph_size, 'seed': opts.seed,
                'physical_parameters': asdict(config),
                'distributions': {
                    'task_coordinates': 'uniform over the square map',
                    'robot_initial_coordinates': 'uniform over the square map',
                    'robot_count': 'uniform choice from robot_counts; n_agents fixes one count',
                    'deadline': 'continuous uniform [deadline_min, deadline_max]',
                    'workload': 'continuous uniform [workload_min, workload_max]',
                    'work_capacity': 'continuous uniform [work_capacity_min, work_capacity_max]',
                },
                'units': {'inputs': 'physical map/work units; deadlines in seconds',
                          'stored': 'normalized model units; deadlines remain seconds'},
            }
            path.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
            print('Saved {} scenarios, {} tasks/scenario, robot counts {}: {}'.format(
                len(dataset), config.graph_size, config.robot_counts, path))
            print('Parameters: {}'.format(path.with_suffix('.json')))
            outputs.append(path)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return outputs


if __name__ == '__main__':
    main()
