"""Plot train/validation CSV curves without parsing console output."""
import argparse
import csv
import math
from pathlib import Path


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics_dir', required=True)
    parser.add_argument('--metric', choices=['mean_cost', 'mean_reward'], default='mean_cost')
    parser.add_argument('--x', choices=['step', 'epoch'], default='step')
    parser.add_argument('--output', help='PNG/PDF/SVG filename, default inside metrics_dir')
    opts = parser.parse_args(args)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(9, 5))
    plotted = 0
    for stream, label in [('train_batch', 'Train batches'), ('train_epoch', 'Train epochs'),
                          ('validation_epoch', 'Validation')]:
        path = Path(opts.metrics_dir) / (stream + '.csv')
        if not path.exists():
            continue
        # Keep resumed/concurrent run names separate if CSV files were appended.
        groups = {}
        with path.open(newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                try:
                    x, y = float(row.get(opts.x) or 'nan'), float(row.get(opts.metric) or 'nan')
                except (ValueError, TypeError):
                    continue
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                groups.setdefault(row['run'], []).append((x, y))
        for run, values in groups.items():
            x, y = zip(*values)
            axis.plot(x, y, label=label if len(groups) == 1 else '{} ({})'.format(label, run),
                      alpha=.3 if stream == 'train_batch' else 1.,
                      linewidth=.7 if stream == 'train_batch' else 1.8,
                      marker=None if stream == 'train_batch' else 'o')
            plotted += 1
    if not plotted:
        plt.close(fig)
        parser.error('No train/validation curve rows found for the selected x axis')
    axis.set_xlabel('Optimizer updates' if opts.x == 'step' else 'Epoch (zero-based)')
    axis.set_ylabel('Mean cost (lower is better)' if opts.metric == 'mean_cost'
                    else 'Mean reward = -cost (higher is better)')
    axis.grid(alpha=.25)
    axis.legend()
    fig.tight_layout()
    output = Path(opts.output) if opts.output else Path(opts.metrics_dir) / (opts.metric + '.png')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=160)
    plt.close(fig)
    print('Saved {}'.format(output))
    return output


if __name__ == '__main__':
    main()
