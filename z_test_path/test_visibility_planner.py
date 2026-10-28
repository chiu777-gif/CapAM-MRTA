"""Analytic routes, obstacle topology, units and planner-specific cache checks."""
import itertools
import math
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from problems.mrta.data import validate_instance
from z_test_path.generate_test_set import ObstacleMap
from z_test_path.path_planner import get_path, make_planner, precompute_navigation
from z_test_path.visibility_planner import VisibilityGraphPlanner


class VisibilityPlannerTests(unittest.TestCase):
    def assert_free_route(self, planner, path, start, goal, distance):
        np.testing.assert_array_equal(path[0], start)
        np.testing.assert_array_equal(path[-1], goal)
        self.assertTrue(np.isfinite(path).all())
        self.assertTrue(((path >= 0) & (path <= 1)).all())
        self.assertTrue(all(planner.segment_is_free(a, b) for a, b in zip(path[:-1], path[1:])))
        self.assertAlmostEqual(distance, np.linalg.norm(np.diff(path, axis=0), axis=1).sum(), places=12)

    def test_single_obstacle_matches_analytic_offset_corner_route(self):
        obstacle_map = ObstacleMap(('...', '.#.', '...'))
        epsilon = 1e-6
        planner = make_planner(obstacle_map, path_planner='visibility_dijkstra', visibility_epsilon=epsilon)
        start, goal = np.array([.1, .5]), np.array([.9, .5])
        path, distance = planner.plan(start, goal)
        expected = 2 * math.hypot(1 / 3 - epsilon - .1, 1 / 6 + epsilon) + 1 / 3 + 2 * epsilon
        self.assertEqual(len(path), 4)
        self.assertAlmostEqual(distance, expected, places=12)
        self.assert_free_route(planner, path, start, goal, distance)
        _, grid_distance = make_planner(obstacle_map, subdivisions=10).plan(start, goal)
        self.assertLess(distance, grid_distance)
        # Adding another query reuses the source tree without altering the route.
        planner.plan(start, [.9, .6])
        again, repeated = planner.plan(start, goal)
        np.testing.assert_array_equal(again, path)
        self.assertEqual(repeated, distance)

    def test_empty_map_straight_lines_and_zero_length_need_no_graph(self):
        planner = make_planner(ObstacleMap(('..', '..')), path_planner='visibility_dijkstra')
        with patch.object(planner, '_build_graph', side_effect=AssertionError('Unnecessary graph construction')):
            for start, goal in [([0., 0.], [1., 1.]), ([.4, .5], [.4, .5])]:
                path, distance = planner.plan(start, goal)
                self.assertEqual(path.shape, (2, 2))
                self.assert_free_route(planner, path, start, goal, distance)
                self.assertAlmostEqual(distance, np.linalg.norm(np.asarray(goal) - start))

    def test_closed_obstacles_invalid_points_and_invalid_epsilon(self):
        obstacle_map = ObstacleMap(('...', '.#.', '...'))
        planner = make_planner(obstacle_map, path_planner='visibility_dijkstra')
        for point in ([1 / 3, .5], [1 / 3, 1 / 3], [.5, .5], [-.1, 0.],
                      [1.01, 0.], [math.nan, 0.], [math.inf, 0.], [.1]):
            with self.subTest(point=point), self.assertRaises(ValueError):
                planner.plan(point, [.1, .1])
        for epsilon in (0., -1e-6, math.nan, math.inf, 1e-20, 1 / 6, .5):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                make_planner(obstacle_map, path_planner='visibility_dijkstra', visibility_epsilon=epsilon)
        with self.assertRaisesRegex(ValueError, 'Unknown path_planner'):
            make_planner(obstacle_map, path_planner='unknown')

    def test_all_small_maps_preserve_free_cell_connectivity(self):
        # An independent four-neighbor flood fill checks touching corners,
        # merged obstacles, border walls and rectangular cells in all 2x3 maps.
        for occupied in itertools.product((False, True), repeat=6):
            rows = tuple(''.join('#' if occupied[r * 3 + c] else '.' for c in range(3)) for r in range(2))
            free = {(r, c) for r in range(2) for c in range(3) if rows[r][c] == '.'}
            if not free:
                continue
            planner = make_planner(ObstacleMap(rows), path_planner='visibility_dijkstra')
            for first, second in itertools.combinations(sorted(free), 2):
                reached, pending = {first}, [first]
                while pending:
                    r, c = pending.pop()
                    for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                        if neighbor in free and neighbor not in reached:
                            reached.add(neighbor)
                            pending.append(neighbor)
                start = np.array([(first[1] + .5) / 3, 1 - (first[0] + .5) / 2])
                goal = np.array([(second[1] + .5) / 3, 1 - (second[0] + .5) / 2])
                with self.subTest(rows=rows, first=first, second=second):
                    if second not in reached:
                        with self.assertRaisesRegex(ValueError, 'cannot connect'):
                            planner.plan(start, goal)
                    else:
                        path, distance = planner.plan(start, goal)
                        self.assert_free_route(planner, path, start, goal, distance)

    def item(self):
        return validate_instance({
            'loc': [[.15, .5], [.85, .5]], 'robots_start_location': [[.15, .8], [.5, .5]],
            'n_agents': 1, 'depot': [[.1, .1]], 'max_speed': 1., 'deadline': [10., 10.],
            'workload': [0., 0.], 'robots_work_capacity': [1., 1.],
        })

    def test_cached_tables_reverse_paths_and_model_unit_scaling(self):
        item, obstacle_map = self.item(), ObstacleMap(('...', '.#.', '...'))
        navigation, _ = precompute_navigation(item, obstacle_map, 1., path_planner='visibility_dijkstra')
        self.assertEqual(navigation['path_planner'], 'visibility_dijkstra')
        self.assertIsNone(navigation['subdivisions'])
        self.assertIsNone(navigation['path_subdivisions'])
        np.testing.assert_array_equal(navigation['robot_to_task'][1], 0.)
        np.testing.assert_array_equal(get_path(navigation, 1, 0), get_path(navigation, 0, 1)[::-1])
        self.assertEqual(get_path(navigation, 0, 0).shape, (2, 2))
        larger = dict(item)
        for name in ('loc', 'robots_start_location'):
            larger[name] = item[name] * 4
        scaled, _ = precompute_navigation(larger, obstacle_map, 4., path_planner='visibility_dijkstra')
        np.testing.assert_allclose(scaled['task_to_task'], navigation['task_to_task'] * 4)
        np.testing.assert_allclose(scaled['robot_to_task'], navigation['robot_to_task'] * 4)
        np.testing.assert_allclose(get_path(scaled, 0, 1), get_path(navigation, 0, 1) * 4)

    def test_cache_isolates_algorithms_and_only_uses_effective_parameters(self):
        item, obstacle_map = self.item(), ObstacleMap(('...', '.#.', '...'))
        with tempfile.TemporaryDirectory() as directory:
            grid, _ = precompute_navigation(item, obstacle_map, 1., 2, directory)
            visible, stats = precompute_navigation(item, obstacle_map, 1., 2, directory,
                                                   path_planner='visibility_dijkstra')
            self.assertFalse(stats['cache_hit'])
            self.assertNotEqual(grid['key'], visible['key'])
            with patch.object(VisibilityGraphPlanner, 'plan', side_effect=AssertionError('Cache ran a search')), \
                    patch.object(VisibilityGraphPlanner, '_build_graph', side_effect=AssertionError('Cache built graph')):
                same, stats = precompute_navigation(item, obstacle_map, 1., 7, directory,
                                                      path_planner='visibility_dijkstra')
            self.assertTrue(stats['cache_hit'])
            self.assertEqual(stats['planning_seconds'], 0.)
            self.assertEqual(same['key'], visible['key'])
            changed, stats = precompute_navigation(item, obstacle_map, 1., 2, directory,
                                                   path_planner='visibility_dijkstra', visibility_epsilon=2e-6)
            self.assertFalse(stats['cache_hit'])
            self.assertNotEqual(changed['key'], visible['key'])
            same_grid, stats = precompute_navigation(item, obstacle_map, 1., 2, directory, visibility_epsilon=2e-6)
            self.assertTrue(stats['cache_hit'])
            self.assertEqual(same_grid['key'], grid['key'])
            rebuilt, stats = precompute_navigation(item, obstacle_map, 1., 2, directory, rebuild=True,
                                                   path_planner='visibility_dijkstra')
            self.assertFalse(stats['cache_hit'])
            np.testing.assert_array_equal(get_path(rebuilt, 0, 1), get_path(visible, 0, 1))


if __name__ == '__main__':
    unittest.main()
