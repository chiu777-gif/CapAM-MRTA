"""Generate versioned uniform scenarios in exactly the training distribution."""
import argparse
from dataclasses import replace
from pathlib import Path
from problems.mrta.config import DEFAULT_CONFIG, DEFAULT_SEED, MRTAConfig, add_environment_arguments
from problems.mrta.data import RandomMRTADataset, save_training_dataset


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_environment_arguments(parser, include_graph=False)
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[DEFAULT_CONFIG.graph_size])
    parser.add_argument('--dataset_size', type=int, default=100)
    parser.add_argument('--filename', help='Explicit output path, requires one graph size')
    parser.add_argument('--data_dir', default='data')
    parser.add_argument('--name', default='uniform')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('-f', action='store_true', help='Overwrite existing files')
    opts = parser.parse_args(args)
    if opts.filename and len(opts.graph_sizes) != 1:
        parser.error('--filename requires exactly one graph size')
    try:
        base_config = MRTAConfig.from_options(opts)
        for size in opts.graph_sizes:
            config = replace(base_config, graph_size=size)
            dataset = RandomMRTADataset(config, opts.dataset_size, opts.seed)
            filename = opts.filename or str(Path(opts.data_dir) / 'mrta' /
                                            '{}_{}_seed{}.pkl'.format(opts.name, size, opts.seed))
            path = save_training_dataset(dataset, filename, overwrite=opts.f)
            print('Saved {} scenarios ({} tasks) to {}'.format(len(dataset), size, path))
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
