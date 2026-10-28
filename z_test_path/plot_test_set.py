"""Plot one generated MRTA scenario with obstacles, tasks, and robots."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np

from problems.mrta.config import DATA_VERSION, MRTAConfig
from problems.mrta.data import validate_instance
from z_test_path.generate_test_set import ObstacleMap


def load_scene(path, index=0):
    with Path(path).open('rb') as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or payload.get('format') != 'capam_mrta':
        raise ValueError('Expected a versioned dataset from generate_test_set.py')
    if payload.get('version') != DATA_VERSION:
        raise ValueError('Unsupported dataset version: {}'.format(payload.get('version')))
    units = payload.get('units', {})
    if units.get('representation') != 'normalized':
        raise ValueError('Expected normalized coordinates in the dataset')
    config = MRTAConfig(**payload['generation'])
    if units.get('coordinate_scale') != config.coordinate_scale:
        raise ValueError('Dataset coordinate_scale differs from its generation parameters')
    map_data = payload.get('obstacle_map')
    if not isinstance(map_data, dict) or 'grid' not in map_data:
        raise ValueError('Dataset has no obstacle_map.grid; use generate_test_set.py to generate it')
    obstacle_map = ObstacleMap(map_data['grid'])
    instances = payload.get('instances', [])
    if not 0 <= index < len(instances):
        raise ValueError('Scenario index {} is out of range; dataset contains {} scenarios '
                         '(indices start at 0)'.format(index, len(instances)))
    return config, obstacle_map, validate_instance(instances[index])


def plot_scene(config, obstacle_map, instance, index=0, units='normalized',
               show_ids=False, show_grid=True, title=None, routes=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    factor = config.coordinate_scale if units == 'physical' else 1.
    extent = config.map_size / config.coordinate_scale * factor
    tasks = instance['loc'].detach().cpu().numpy() * factor
    n_agents = instance['n_agents'].item()
    robots = instance['robots_start_location'][:n_agents].detach().cpu().numpy() * factor
    obstacles = np.array([[cell == '#' for cell in row] for row in obstacle_map.grid])
    robot_colors = plt.get_cmap('tab10')

    fig, axis = plt.subplots(figsize=(8, 8), facecolor='white')
    axis.imshow(obstacles, origin='upper', extent=(0, extent, 0, extent),
                cmap=ListedColormap(['#f8fafc', '#334155']), vmin=0, vmax=1,
                interpolation='nearest', zorder=0)
    if show_grid:
        axis.vlines(np.linspace(0, extent, obstacle_map.width + 1), 0, extent,
                    colors='#94a3b8', linewidth=.5, alpha=.35, zorder=1)
        axis.hlines(np.linspace(0, extent, obstacle_map.height + 1), 0, extent,
                    colors='#94a3b8', linewidth=.5, alpha=.35, zorder=1)
    if routes is not None:
        for leg in routes:
            path = np.asarray(leg['path'], dtype=float) * factor
            axis.plot(path[:, 0], path[:, 1], color=robot_colors((leg['robot_id'] - 1) % 10),
                      linewidth=1.3, alpha=.8, zorder=2)
    task_points = axis.scatter(tasks[:, 0], tasks[:, 1], s=28, c='#2563eb',
                               edgecolors='white', linewidths=.4, zorder=3,
                               label='Tasks ({})'.format(len(tasks)))
    colors = [robot_colors(robot % 10) for robot in range(n_agents)] if routes is not None else '#f97316'
    robot_points = axis.scatter(robots[:, 0], robots[:, 1], s=125, c=colors, marker='^',
                                edgecolors='#7c2d12', linewidths=.8, zorder=5,
                                label='Robots ({})'.format(n_agents))
    for number, (x, y) in enumerate(robots, 1):
        axis.annotate('R{}'.format(number), (x, y), xytext=(6, 5),
                      textcoords='offset points', fontsize=9, color='#9a3412', zorder=7)
    if show_ids:
        for number, (x, y) in enumerate(tasks, 1):
            axis.annotate(str(number), (x, y), xytext=(3, 3), textcoords='offset points',
                          fontsize=6, color='#1e40af', zorder=4)

    handles = [task_points, robot_points]
    if obstacles.any():
        handles.append(Patch(facecolor='#334155', label='Obstacle'))
    if routes is not None:
        handles.append(Line2D([0], [0], color='#64748b', label='Robot routes'))
    axis.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, -.09),
                ncol=len(handles), frameon=False, fontsize=9)
    axis.set(xlim=(0, extent), ylim=(0, extent), xlabel='x ({})'.format(units),
             ylabel='y ({})'.format(units))
    axis.set_aspect('equal', adjustable='box')
    subtitle = 'Scenario {} | {} tasks, {} robots'.format(index, len(tasks), n_agents)
    axis.set_title('{}\n{}'.format(title, subtitle) if title else subtitle, fontsize=12, pad=12)
    for spine in axis.spines.values():
        spine.set_color('#64748b')
    fig.tight_layout()
    return fig, axis


def load_routes(path, dataset_path, index):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if payload.get('format') != 'mrta_obstacle_routes' or payload.get('version') != 1:
        raise ValueError('Unsupported route file format')
    if payload.get('coordinate_units') != 'normalized':
        raise ValueError('Expected normalized route coordinates')
    if payload.get('dataset_sha256') != hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest():
        raise ValueError('Route file belongs to a different dataset; regenerate routes after changing data')
    for scenario in payload['scenarios']:
        if scenario['index'] == index:
            return scenario['legs']
    raise ValueError('Route file does not contain scenario index {}'.format(index))


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', '--dataset', type=Path, required=True,
                        help='Generated obstacle dataset (.pkl)')
    parser.add_argument('--index', type=int, default=0, help='Scenario index, starting at 0')
    parser.add_argument('--routes', type=Path, help='routes.json produced by the obstacle evaluator')
    parser.add_argument('--units', choices=['normalized', 'physical'], default='normalized')
    parser.add_argument('--show_ids', action='store_true', help='Label tasks with IDs starting at 1')
    parser.add_argument('--no_grid', action='store_true', help='Hide cell boundary lines')
    parser.add_argument('--output', '-o', type=Path, help='Output PNG, PDF, or SVG filename')
    parser.add_argument('--dpi', type=int, default=180, help='Raster image resolution')
    parser.add_argument('--overwrite', '-f', action='store_true')
    opts = parser.parse_args(args)
    if opts.index < 0 or opts.dpi < 1:
        parser.error('index must be nonnegative and dpi must be positive')
    output = opts.output or Path(__file__).resolve().parent / 'figures' / '{}_scene{}{}.png'.format(
        opts.data.stem, opts.index, '_routes' if opts.routes else '')
    if output.suffix.lower() not in ('.png', '.pdf', '.svg'):
        parser.error('output must end in .png, .pdf, or .svg')

    try:
        if output.exists() and not opts.overwrite:
            raise FileExistsError('Already exists: {}. Use --overwrite to replace it.'.format(output))
        config, obstacle_map, instance = load_scene(opts.data, opts.index)
        routes = load_routes(opts.routes, opts.data, opts.index) if opts.routes else None
        fig, _ = plot_scene(config, obstacle_map, instance, index=opts.index, units=opts.units,
                            show_ids=opts.show_ids, show_grid=not opts.no_grid, title=opts.data.stem, routes=routes)
        import matplotlib.pyplot as plt
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(output), dpi=opts.dpi, bbox_inches='tight', facecolor='white')
        finally:
            plt.close(fig)
    except ImportError:
        parser.error('Plotting requires matplotlib; run in an environment with matplotlib installed')
    except (ValueError, OSError, KeyError, TypeError, pickle.UnpicklingError, EOFError) as exc:
        parser.error(str(exc))
    print('Saved {}'.format(output))
    return output


if __name__ == '__main__':
    main()
