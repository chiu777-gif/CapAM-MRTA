"""Export the two protocols' robot-count comparisons as PNG and SVG."""
import argparse
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.cache' / 'matplotlib'))
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=HERE / 'results/n100_train_matched_1000_seed20261024')
    opts = parser.parse_args()
    names = {'upstream_taptc': ('Authors\' TAPTC simulator', '#7c3aed', '--', 's'),
             'capam_full_visit': ('Full-visit adapter', '#0f766e', '-', 'o')}
    summaries = {protocol: json.loads((opts.results / protocol / 'summary.json').read_text())
                 for protocol in names}
    first, second = summaries.values()
    for key in ('dataset_sha256', 'offset', 'num_samples_requested', 'robot_count_histogram', 'cpu_threads'):
        if first[key] != second[key]:
            raise ValueError('Incompatible protocol summaries: ' + key)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for protocol, (label, color, style, marker) in names.items():
        summary = summaries[protocol]
        counts = sorted(map(int, summary['by_robot_count']))
        groups = [summary['by_robot_count'][str(count)] for count in counts]
        for axis, key, scale in ((axes[0], 'mean_cost', 1),
                                 (axes[1], 'completion_percent', 1),
                                 (axes[2], 'inference_seconds_per_scenario', 1000)):
            if any(group[key] is None for group in groups):
                continue
            axis.plot(counts, [group[key] * scale for group in groups],
                      label=label, color=color, linestyle=style, marker=marker,
                      linewidth=2, markersize=6)
    for axis, title, ylabel in zip(axes,
                                  ('Deadline cost', 'On-time task completion', 'Full-solution CPU latency'),
                                  ('Mean cost per scenario', 'Completed tasks (%)', 'Mean milliseconds per scenario')):
        axis.set(title=title, xlabel='Number of robots', ylabel=ylabel)
        axis.set_xticks(counts)
        axis.grid(alpha=.2)
        axis.spines[['top', 'right']].set_visible(False)
        axis.legend(frameon=False, fontsize=8, loc='best')
        axis.set_ylim(bottom=0)
    axes[1].set_ylim(0, 103)
    axes[0].text(.04, .06, 'Original complete-visit cost is undefined\nwhen tasks remain unassigned.',
                 transform=axes[0].transAxes, fontsize=8, color='#6b7280')
    fig.suptitle('BiG-MRTA | {} scenarios, 100 tasks each | CPU, {} threads, batch size 1'.format(
        first['scenarios'], first['cpu_threads']), fontsize=13, fontweight='bold')
    for extension in ('png', 'svg'):
        path = opts.results / ('metrics_by_robot_count.' + extension)
        fig.savefig(path, dpi=180, bbox_inches='tight')
        print(path.resolve())
    plt.close(fig)


if __name__ == '__main__':
    main()
