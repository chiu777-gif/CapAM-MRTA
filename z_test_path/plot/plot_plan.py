"""Draw a selected scenario's evaluated robot routes with direction arrows."""
import argparse
from collections import Counter
from pathlib import Path
import pickle
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np

from z_test_path.plot_test_set import load_routes, load_scene, plot_scene

BASE = REPO / 'z_test_path'


def draw_plan(config, obstacle_map, instance, routes, index=0, units='normalized',
              show_ids=False, show_grid=True, title=None):
    """Reuse the scenario renderer; add movement direction and a robot legend."""
    robots, tasks = int(instance['n_agents']), len(instance['loc'])
    for leg in routes:
        if not 1 <= leg['robot_id'] <= robots or not 1 <= leg['task_id'] <= tasks:
            raise ValueError('Route contains a robot/task ID outside the selected scenario')
        path = np.asarray(leg['path'], dtype=float)
        if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2 or not np.isfinite(path).all():
            raise ValueError('Each route leg must contain at least two finite (x, y) points')

    fig, axis = plot_scene(config, obstacle_map, instance, index=index, units=units,
                           show_ids=show_ids, show_grid=show_grid, title=title, routes=routes)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    factor = config.coordinate_scale if units == 'physical' else 1.
    extent = config.map_size / config.coordinate_scale * factor
    colors = plt.get_cmap('tab10')
    for leg in routes:
        path = np.asarray(leg['path'], dtype=float) * factor
        lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
        segment = int(lengths.argmax())
        if lengths[segment] == 0:
            continue
        # Put the arrow entirely on one segment, so it cannot cut a bend.
        middle = (path[segment] + path[segment + 1]) / 2
        direction = (path[segment + 1] - path[segment]) / lengths[segment]
        half_length = min(lengths[segment] * .2, extent * .014)
        axis.annotate('', xy=middle + direction * half_length,
                      xytext=middle - direction * half_length, zorder=4,
                      arrowprops=dict(arrowstyle='-|>', mutation_scale=9, linewidth=1.,
                                      color=colors((leg['robot_id'] - 1) % 10),
                                      shrinkA=0, shrinkB=0, alpha=.9))

    counts = Counter(leg['robot_id'] for leg in routes)
    handles = [Line2D([], [], color='#2563eb', marker='o', linestyle='none',
                      markersize=5, label='Tasks ({})'.format(tasks))]
    if any('#' in row for row in obstacle_map.grid):
        handles.append(Patch(facecolor='#334155', label='Obstacle'))
    for robot in range(1, robots + 1):
        handles.append(Line2D([], [], color=colors((robot - 1) % 10), marker='^',
                              markersize=6, linewidth=1.4,
                              label='R{} ({} tasks)'.format(robot, counts[robot])))
    axis.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, -.09),
                ncol=min(3, len(handles)), frameon=False, fontsize=8)
    fig.tight_layout()
    return fig, axis


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', '--dataset', type=Path,
                        default=BASE / 'data/n100_obstacles_example.pkl')
    parser.add_argument('--routes', type=Path, default=BASE / 'results/capam_path_100/routes.json',
                        help='routes.json written by z_test_path/evaluate.py')
    parser.add_argument('--index', type=int, default=0, help='Dataset scenario index, starting at 0')
    parser.add_argument('--units', choices=('normalized', 'physical'), default='normalized')
    parser.add_argument('--show_ids', action='store_true', help='Label task IDs starting at 1')
    parser.add_argument('--no_grid', action='store_true')
    parser.add_argument('--output', '-o', type=Path, help='PNG, PDF, or SVG output file')
    parser.add_argument('--dpi', type=int, default=180)
    parser.add_argument('--overwrite', '-f', action='store_true')
    opts = parser.parse_args(args)
    if opts.index < 0 or opts.dpi < 1:
        parser.error('index must be nonnegative and dpi must be positive')
    output = opts.output or Path(__file__).resolve().parent / 'figures' / opts.routes.parent.name / (
        '{}_scene{}_plan.png'.format(opts.data.stem, opts.index))
    if output.suffix.lower() not in ('.png', '.pdf', '.svg'):
        parser.error('output must end in .png, .pdf, or .svg')

    try:
        if output.exists() and not opts.overwrite:
            raise FileExistsError('Already exists: {}. Use --overwrite to replace it.'.format(output))
        config, obstacle_map, instance = load_scene(opts.data, opts.index)
        routes = load_routes(opts.routes, opts.data, opts.index)
        fig, _ = draw_plan(config, obstacle_map, instance, routes, index=opts.index, units=opts.units,
                           show_ids=opts.show_ids, show_grid=not opts.no_grid,
                           title='{} | Planned routes'.format(opts.routes.parent.name))
        import matplotlib.pyplot as plt
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(output), dpi=opts.dpi, bbox_inches='tight', facecolor='white')
        finally:
            plt.close(fig)
    except ImportError:
        parser.error('Plotting requires matplotlib; run in an environment with matplotlib installed')
    except (ValueError, OSError, KeyError, TypeError, pickle.UnpicklingError, EOFError) as error:
        parser.error(str(error))
    print('Saved {} (index={}, {} route legs)'.format(output, opts.index, len(routes)))
    return output


if __name__ == '__main__':
    main()
