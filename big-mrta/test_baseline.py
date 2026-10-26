"""Behavioral checks against the original driver and the shared MRTA replay."""
import ast
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from baseline import HERE, matrices, prepare, replay_full_visit, solve_full_visit, solve_upstream
import numpy as np
import pandas as pd
import torch
from problems.mrta.config import DEFAULT_CONFIG
from problems.mrta.data import load_file_dataset, normalize_instance
from evaluate import dataset_fingerprint, upstream_manifest


def physical_instance(loc, starts, deadlines, workload=None):
    return normalize_instance(dict(
        loc=loc, robots_start_location=starts, deadline=deadlines,
        workload=workload if workload is not None else [1.] * len(loc),
        robots_work_capacity=[1.] * len(starts), n_agents=len(starts),
    ), DEFAULT_CONFIG)


def original_driver(scenario):
    """Execute the actual vendored script, changing only inputs/output location.

    This is an independent oracle for the hand-ported native simulator. No
    changes to the policy, time loop, resource updates or matching are made.
    """
    source = (HERE / 'upstream/main_bigmrta_TAPTC.py').read_text()
    tasks = pd.DataFrame(np.column_stack((np.arange(scenario.n_tasks), scenario.locations,
                                         scenario.workload, scenario.deadlines)))
    robots = pd.DataFrame(np.column_stack((np.arange(scenario.n_robots), scenario.starts,
                                          scenario.capacities)))
    with tempfile.TemporaryDirectory() as tmp:
        replacements = dict(output_dir=tmp, maxRun=1, group_list=[1],
                            instance_list=[0], ratio_deadline_list=[1],
                            robotSize_list=[scenario.n_robots], velVal=scenario.speed)

        class Inputs(ast.NodeTransformer):
            def visit_Assign(self, node):
                if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                    if name in replacements:
                        node.value = ast.parse(repr(replacements[name]), mode='eval').body
                return self.generic_visit(node)

        tree = ast.fix_missing_locations(Inputs().visit(ast.parse(source)))
        namespace = {'__name__': '__baseline_oracle__'}
        with patch('pandas.read_csv', side_effect=[tasks.copy(), robots.copy()]), contextlib.redirect_stdout(io.StringIO()):
            try:
                exec(compile(tree, 'upstream/main_bigmrta_TAPTC.py', 'exec'), namespace)
            except SystemExit:
                pass
        return namespace


class BiGMRTATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.dataset = load_file_dataset(HERE.parent / 'z_test/data/n100_train_matched_1000_seed20261024.pkl', num_samples=3)

    def test_vendored_source_checksums(self):
        self.assertEqual(upstream_manifest()['commit'], 'e49cd948653b15a70348d85a65b1b42c847f8b5b')

    def test_saved_units_and_active_robots_are_preserved(self):
        item = self.dataset[0]
        scenario = prepare(item)
        self.assertEqual(scenario.n_robots, int(item['n_agents']))
        np.testing.assert_array_equal(scenario.locations, item['loc'].numpy().astype(np.float64) * 100)
        np.testing.assert_array_equal(scenario.deadlines, item['deadline'].numpy())
        self.assertAlmostEqual(scenario.speed, float(item['max_speed']) * 100)
        distance, times = matrices(scenario, ceil_service=False)
        first = np.linalg.norm(scenario.starts[0] - scenario.locations[0]) / scenario.speed
        self.assertAlmostEqual(times['robot-1'][0, 1], first + scenario.workload[0] / scenario.capacities[0])
        self.assertAlmostEqual(distance[0, 1], np.linalg.norm(scenario.starts[-1] - scenario.locations[0]))

    def compare_native_to_oracle(self, scenario):
        native = solve_upstream(scenario)
        oracle = original_driver(scenario)
        # Column 6 contains wall-clock policy timings and cannot be identical.
        columns = [0, 1, 2, 3, 4, 5, 7, 8]
        np.testing.assert_array_equal(native['robot_state'][:, columns], oracle['robotState'][:, columns])
        expected = [(float(row[0]), int(row[4]), int(row[3]))
                    for row in oracle['decisionHistory'][1:] if row[3] != 0]
        actual = [(event['decision_time'], event['robot'], event['task']) for event in native['events']]
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), oracle['numTaskDone'])
        self.assertEqual(native['upstream_travel_cost'], oracle['totalCost'])

    def test_native_driver_matches_actual_original_script(self):
        for item in self.dataset:
            with self.subTest(n_agents=int(item['n_agents'])):
                self.compare_native_to_oracle(prepare(item))

    def test_original_taptc_text_keeps_exact_service_times(self):
        data = load_file_dataset(HERE.parent / 'data/mrta_gini')
        self.assertEqual(len(data), 96)
        for index in (0, 3, 6, 9):
            scenario = prepare(data[index], taptc_source=data.names[index])
            task_path = Path(data.names[index])
            task_data = np.loadtxt(task_path, skiprows=1, ndmin=2)
            agent_data = np.loadtxt(task_path.parent.parent / 'agent' /
                                    ('a' + task_path.stem.split('a', 1)[1] + '.txt'),
                                    skiprows=1, ndmin=2)
            self.assertEqual(scenario.speed, 1.)
            np.testing.assert_array_equal(scenario.workload, task_data[:, 3])
            np.testing.assert_array_equal(scenario.capacities, agent_data[:, 3])
            _, times = matrices(scenario, ceil_service=True)
            for robot in range(scenario.n_robots):
                travel = np.linalg.norm(scenario.locations - scenario.starts[robot], axis=1)
                expected = travel + np.ceil(task_data[:, 3] / agent_data[robot, 3])
                np.testing.assert_array_equal(times['robot-{}'.format(robot + 1)][0, 1:], expected)
            self.compare_native_to_oracle(scenario)
            replay_full_visit(scenario, solve_full_visit(scenario))

    def test_directory_fingerprint_covers_tasks_and_paired_agents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'group1').mkdir()
            (root / 'agent').mkdir()
            task = root / 'group1/r11a2i00.txt'
            agent = root / 'agent/a2i00.txt'
            task.write_text('task input')
            agent.write_text('robot input')
            first = dataset_fingerprint(root)
            self.assertEqual([item['path'] for item in first['files']],
                             ['agent/a2i00.txt', 'group1/r11a2i00.txt'])
            (root / 'n_200').mkdir()
            (root / 'n_200/extra.txt').write_text('not an evaluated input')
            self.assertEqual(first, dataset_fingerprint(root))
            agent.write_text('changed robot input')
            self.assertNotEqual(first['sha256'], dataset_fingerprint(root)['sha256'])
            pickle = root / 'generated.pkl'
            pickle.write_bytes(b'fixed generated dataset')
            import hashlib
            self.assertEqual(dataset_fingerprint(pickle)['sha256'],
                             hashlib.sha256(pickle.read_bytes()).hexdigest())

    def test_native_idle_return_and_sparse_matching_match_original(self):
        scenario = prepare(physical_instance(
            [[1., 0.], [99., 100.], [50., 50.], [0., 3.], [90., 100.]],
            [[0., 0.], [100., 100.]], [2., 2., 5., 10., 20.]))
        self.compare_native_to_oracle(scenario)
        self.assertTrue(np.isinf(solve_upstream(scenario)['finish_times']).any())

    def test_full_visit_handles_impossible_tasks_and_replays(self):
        scenario = prepare(physical_instance([[100., 100.], [90., 90.]], [[0., 0.]], [1., 2.]))
        native = solve_upstream(scenario)
        self.assertTrue(np.isinf(native['finish_times']).all())
        result = solve_full_visit(scenario)
        self.assertEqual(sorted(event['task'] for event in result['events']), [1, 2])
        self.assertEqual(result['relaxed_decisions'], 2)
        cost, done = replay_full_visit(scenario, result)
        self.assertGreater(cost, 0)
        self.assertEqual(done, 0)

    def test_deadline_equality_is_completed(self):
        # Stored workload and capacity have identical float32 scaling, so time=1.
        scenario = prepare(physical_instance([[0., 0.]], [[0., 0.]], [1.]))
        cost, done = replay_full_visit(scenario, solve_full_visit(scenario))
        self.assertEqual(cost, 0.)
        self.assertEqual(done, 1)

    def test_fewer_tasks_than_robots_still_allocates_every_task(self):
        scenario = prepare(physical_instance([[3., 4.]], [[0., 0.]] * 7, [1.]))
        result = solve_full_visit(scenario)
        self.assertEqual(len(result['events']), 1)
        self.assertEqual(result['events'][0]['robot'], 0)
        self.assertEqual(replay_full_visit(scenario, result)[1], 0)

    def test_full_dataset_samples_use_exact_common_replay(self):
        for item in self.dataset:
            scenario = prepare(item)
            first = solve_full_visit(scenario)
            second = solve_full_visit(scenario)
            self.assertEqual(first['events'], second['events'])
            self.assertEqual(len(first['events']), scenario.n_tasks)
            replay_full_visit(scenario, first)


if __name__ == '__main__':
    unittest.main()
