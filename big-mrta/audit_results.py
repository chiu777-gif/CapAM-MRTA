"""Validate exported routes/metrics against the fixed dataset without timing."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from baseline import HERE
from evaluate import dataset_fingerprint, summarize, upstream_manifest
import numpy as np
from problems.mrta.data import load_file_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=HERE / 'results/n100_train_matched_1000_seed20261024')
    opts = parser.parse_args()
    manifest = upstream_manifest()
    audit = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(), protocols={})
    for protocol in ('upstream_taptc', 'capam_full_visit'):
        directory = opts.results / protocol
        summary = json.loads((directory / 'summary.json').read_text())
        assert summary['upstream'] == manifest
        data_path = Path(summary['data_sources'][0])
        assert dataset_fingerprint(data_path)['sha256'] == summary['dataset_sha256']
        data = load_file_dataset(data_path, offset=summary['offset'], num_samples=summary['num_samples_requested'])
        with (directory / 'scenarios.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        with (directory / 'routes.jsonl').open() as handle:
            routes = [json.loads(line) for line in handle]
        assert len(rows) == len(routes) == len(data) == summary['scenarios']
        typed_rows = []
        for index, (row, route, instance) in enumerate(zip(rows, routes, data)):
            assert row['instance'] == data.names[index]
            assert int(row['scenario_index']) == route['scenario_index'] == summary['offset'] + index
            assert int(row['n_agents']) == route['n_agents'] == int(instance['n_agents'])
            deadline = instance['deadline'].numpy().astype(np.float64)
            assert len(route['finish_times']) == len(deadline) == int(row['task_count'])
            finish = np.array([np.inf if value is None else value for value in route['finish_times']])
            assigned = np.isfinite(finish)
            done = assigned & (finish <= deadline)
            late = assigned & (finish > deadline)
            ids = [event['task'] for event in route['events']]
            assert len(set(ids)) == len(ids) == int(assigned.sum())
            assert set(ids) == set((np.flatnonzero(assigned) + 1).tolist())
            for event in route['events']:
                assert 0 <= event['robot'] < int(instance['n_agents'])
                assert event['finish_time'] == finish[event['task'] - 1]
            assigned_cost = float((finish[late] / deadline[late]).sum())
            cost = float(row['cost']) if row['cost'] else None
            assert (cost is not None) == bool(assigned.all())
            if cost is not None:
                assert np.isclose(cost, assigned_cost, rtol=2e-6, atol=2e-5)
            if protocol == 'capam_full_visit':
                assert sorted(ids) == list(range(1, len(deadline) + 1))
            assert int(done.sum()) == int(row['tasks_done'])
            assert int(late.sum()) == int(row['late_assigned_tasks'])
            assert int((~assigned).sum()) == int(row['unassigned_tasks'])
            assert np.isclose(float(row['assigned_task_deadline_cost']), assigned_cost)
            assert np.isclose(float(row['completion_rate']), done.mean())
            assert float(row['policy_seconds']) <= float(row['inference_seconds'])
            assert sum(event['deadline_relaxed'] for event in route['events']) == int(row['relaxed_decisions'])
            typed_rows.append({key: (cost if key == 'cost' else float(value))
                               for key, value in row.items()
                               if key in ('task_count', 'tasks_done', 'assigned_tasks', 'unassigned_tasks',
                                          'late_assigned_tasks', 'completion_rate', 'cost',
                                          'assigned_task_deadline_cost', 'inference_seconds', 'policy_seconds',
                                          'policy_calls', 'relaxed_decisions', 'isolated_relaxed_decisions')})
        for key, value in summarize(typed_rows).items():
            if value is None:
                assert summary[key] is None
            else:
                assert np.isclose(summary[key], value), key
        counts = dict(sorted(Counter(str(int(row['n_agents'])) for row in rows).items()))
        assert counts == summary['robot_count_histogram']
        audit['protocols'][protocol] = dict(
            scenarios_verified=len(rows), robot_count_histogram=counts,
            completed_tasks=summary['completed_tasks'], unassigned_tasks=summary['unassigned_tasks'],
            mean_cost=summary['mean_cost'], dataset_sha256=summary['dataset_sha256'],
            checks=['source and dataset hashes', 'scenario identities and active robot counts',
                    'unique task assignments and recorded finish times', 'deadline completion and cost',
                    'relaxation flags and policy/full inference timing', 'overall aggregates'],
        )
        print('{}: {} exported scenarios verified'.format(protocol, len(rows)))
    (opts.results / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')


if __name__ == '__main__':
    main()
