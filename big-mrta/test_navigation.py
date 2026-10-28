"""Detours must reach policy matrices, simulation and replay, including cache hits."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from baseline import matrices, prepare, replay_full_visit, solve_full_visit, solve_upstream
from navigation import plan, plan_scenarios
from test_baseline import physical_instance
import numpy as np
import torch
from problems.mrta.config import DEFAULT_CONFIG
from z_test_path.generate_test_set import ObstacleMap
from z_test_path.path_planner import precompute_navigation


class NavigationTests(unittest.TestCase):
    def test_parallel_cold_and_warm_tables_match_serial_in_input_order(self):
        obstacle_map = ObstacleMap(('.....', '..#..', '..#..', '..#..', '.....'))
        instances = [physical_instance([[80. + i, 50.], [85., 60.]], [[20., 50.]], [65., 200.])
                     for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            serial = list(plan_scenarios(instances, obstacle_map, DEFAULT_CONFIG,
                                        Path(tmp) / 'serial', 1e-6, workers=1))
            parallel = list(plan_scenarios(instances, obstacle_map, DEFAULT_CONFIG,
                                          Path(tmp) / 'parallel', 1e-6, workers=2))
            warm = list(plan_scenarios(instances, obstacle_map, DEFAULT_CONFIG,
                                      Path(tmp) / 'parallel', 1e-6, workers=2))
            self.assertEqual([row[0] for row in parallel], [0, 1, 2])
            for left, right, hit in zip(serial, parallel, warm):
                self.assertFalse(right[2]['cache_hit'])
                self.assertGreater(right[2]['path_planning_seconds'], 0.)
                self.assertTrue(hit[2]['cache_hit'])
                self.assertEqual(hit[2]['path_planning_seconds'], 0.)
                self.assertEqual(left[2]['navigation_key'], right[2]['navigation_key'])
                self.assertEqual(left[2]['navigation_key'], hit[2]['navigation_key'])
                for name in ('task_to_task', 'robot_to_task'):
                    np.testing.assert_array_equal(left[1][name], right[1][name])
                    np.testing.assert_array_equal(left[1][name], hit[1][name])
        with self.assertRaises(ValueError):
            list(plan_scenarios(instances, obstacle_map, DEFAULT_CONFIG, None, 1e-6, workers=0))

    def test_detour_cache_policy_and_both_simulators(self):
        torch.set_num_threads(1)
        item = physical_instance([[80., 50.], [85., 60.]], [[20., 50.]], [65., 200.])
        obstacle_map = ObstacleMap(('.....', '..#..', '..#..', '..#..', '.....'))
        with tempfile.TemporaryDirectory() as tmp:
            tables, cold = plan(item, obstacle_map, DEFAULT_CONFIG, Path(tmp), 1e-6)
            self.assertFalse(cold['cache_hit'])
            self.assertGreater(cold['path_planning_seconds'], 0)
            # A warm run must never construct/search a visibility graph.
            with patch('z_test_path.visibility_planner.VisibilityGraphPlanner.plan', side_effect=AssertionError('replanning')):
                warm_tables, warm = plan(item, obstacle_map, DEFAULT_CONFIG, Path(tmp), 1e-6)
            self.assertTrue(warm['cache_hit'])
            self.assertEqual(warm['path_planning_seconds'], 0)
            self.assertEqual(cold['navigation_key'], warm['navigation_key'])
            # This is exactly the z_test_path cache, not a BiG-specific copy.
            shared, stats = precompute_navigation(item, obstacle_map, 1., cache_dir=tmp,
                                                  path_planner='visibility_dijkstra')
            self.assertTrue(stats['cache_hit'])
            np.testing.assert_array_equal(tables['task_to_task'], shared['task_to_task'])
            direct = prepare(item)
            scenario = prepare(item, navigation=tables)
            distance, times = matrices(scenario, ceil_service=False)
            self.assertGreater(distance[0, 1], 60.)
            self.assertAlmostEqual(times['robot-1'][0, 1], distance[0, 1] / scenario.speed + 1.)
            self.assertEqual(replay_full_visit(direct, solve_full_visit(direct))[1], 2)
            result = solve_full_visit(scenario)
            cost, done = replay_full_visit(scenario, result)
            self.assertGreater(cost, 0.)
            self.assertEqual(done, 1)
            self.assertGreater(solve_upstream(scenario)['finish_times'][0], 65.)
            warm_result = solve_full_visit(prepare(item, navigation=warm_tables))
            self.assertEqual(result['events'], warm_result['events'])
            cost_result = solve_full_visit(scenario, objective='cost')
            replay_full_visit(scenario, cost_result)
            self.assertEqual(cost_result['events'],
                             solve_full_visit(prepare(item, navigation=warm_tables), objective='cost')['events'])
            _, changed = plan(item, obstacle_map, DEFAULT_CONFIG, Path(tmp), .001)
            self.assertFalse(changed['cache_hit'])
            self.assertNotEqual(changed['navigation_key'], cold['navigation_key'])


if __name__ == '__main__':
    unittest.main()
