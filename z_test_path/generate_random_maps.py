"""Generate reproducible ASCII maps while preserving four-neighbor free connectivity."""
import argparse
from collections import deque
import json
from pathlib import Path
import random


def free_connected(grid):
    size = len(grid)
    free = {(r, c) for r in range(size) for c in range(size) if grid[r][c] == '.'}
    if not free:
        return False
    seen = {min(free)}
    queue = deque(seen)
    while queue:
        r, c = queue.popleft()
        for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if point in free and point not in seen:
                seen.add(point)
                queue.append(point)
    return len(seen) == len(free)


def generate_map(size, density_percent, style, seed):
    rng = random.Random(seed)
    grid = [['.'] * size for _ in range(size)]
    target = round(size * size * density_percent / 100)
    blocked = 0
    for _ in range(50000):
        if blocked == target:
            return [''.join(row) for row in grid]
        height = width = 1
        if style == 'blocks':
            height, width = rng.randint(1, min(4, size)), rng.randint(1, min(4, size))
        r, c = rng.randrange(size - height + 1), rng.randrange(size - width + 1)
        changed = [(i, j) for i in range(r, r + height) for j in range(c, c + width)
                   if grid[i][j] == '.']
        if not changed or blocked + len(changed) > target:
            continue
        for i, j in changed:
            grid[i][j] = '#'
        if free_connected(grid):
            blocked += len(changed)
        else:
            for i, j in changed:
                grid[i][j] = '.'
    raise ValueError('Could not reach {}% for {} seed {}; try another seed or lower density'.format(
        density_percent, style, seed))


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, default=Path(__file__).resolve().parent / 'maps')
    parser.add_argument('--size', type=int, default=20, help='Rows and columns')
    parser.add_argument('--densities', type=int, nargs='+', default=[10, 20, 30], help='Obstacle percentages')
    parser.add_argument('--styles', choices=['cells', 'blocks'], nargs='+', default=['cells', 'blocks'])
    parser.add_argument('--count', type=int, default=2, help='Maps per style/density')
    parser.add_argument('--seed', type=int, default=20261027, help='Each variant uses seed + variant index')
    parser.add_argument('--overwrite', action='store_true')
    opts = parser.parse_args(args)
    if opts.size < 2 or opts.count < 1 or opts.seed < 0:
        parser.error('size must be >= 2, count positive and seed nonnegative')
    if any(not 0 < value < 100 for value in opts.densities):
        parser.error('densities must be integer percentages in (0, 100)')
    if len(set(opts.densities)) != len(opts.densities) or len(set(opts.styles)) != len(opts.styles):
        parser.error('densities and styles must not contain duplicates')
    manifest_name = 'random_maps_seed{}.json'.format(opts.seed)
    jobs = [(style, density, opts.seed + index) for style in opts.styles
            for density in opts.densities for index in range(opts.count)]
    filenames = ['random_{}_d{:02d}_s{}.txt'.format(*job) for job in jobs]
    for name in filenames + [manifest_name]:
        if (opts.output_dir / name).exists() and not opts.overwrite:
            parser.error('Already exists: {}. Choose another output directory/seed or use --overwrite'.format(
                opts.output_dir / name))
    # Finish the whole batch before writing files, including density feasibility checks.
    maps, entries = [], []
    for (style, density, seed), name in zip(jobs, filenames):
        try:
            rows = generate_map(opts.size, density, style, seed)
        except ValueError as error:
            parser.error(str(error))
        count = sum(row.count('#') for row in rows)
        maps.append(rows)
        entries.append(dict(file=name, style=style, seed=seed, rows=opts.size, columns=opts.size,
                            requested_density_percent=density, obstacle_cells=count,
                            free_cells=opts.size ** 2 - count,
                            density_percent=100 * count / opts.size ** 2,
                            free_region_four_connected=True))
    manifest = dict(generator='z_test_path/generate_random_maps.py', generator_version=1,
                    algorithm='Incremental random cell/rectangle proposals; reject additions that disconnect free cells',
                    rng='Python random.Random (MT19937)', base_seed=opts.seed,
                    seed_rule='seed + variant index, shared across styles and densities', maps=entries)
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    for entry, rows in zip(entries, maps):
        (opts.output_dir / entry['file']).write_text('\n'.join(rows) + '\n', encoding='utf-8')
        print('{}: {}x{}, {:.1f}% obstacles, seed={}'.format(
            entry['file'], opts.size, opts.size, entry['density_percent'], entry['seed']))
    (opts.output_dir / manifest_name).write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    main()
