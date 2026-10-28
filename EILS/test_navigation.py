"""Obstacle detours, both search backends, cache reuse and saved result replay."""
import contextlib
from dataclasses import asdict
import io
from pathlib import Path
import pickle
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from EILS.compare_backends import compare, read_routes, read_rows
from EILS.evaluate import main as evaluate
from EILS.navigation import load_evaluation_data
from EILS.solver import prepare, solve, validate_solution
from EILS.test_solver import instance
from EILS.verify_results import verify
from problems.mrta.config import DATA_VERSION, MRTAConfig
from problems.mrta.data import normalize_instance
from z_test_path.path_planner import get_path, make_planner, precompute_navigation


HERE = Path(__file__).resolve().parent


def obstacle_fixture(directory):
    # Extent is 2 rather than 1; the inactive robot lies inside the obstacle.
    config = MRTAConfig(graph_size=3, robot_counts=(1,), map_size=200)
    instances = []
    for index in range(3):
        instances.append(normalize_instance(dict(
            loc=np.array([[.8, .5], [.9, .8], [.1, .8]]) * config.map_size,
            robots_start_location=np.array([[.2 + .01 * index, .5], [.5, .5]]) * config.map_size,
            deadline=[130, 100, 100], workload=[1, 1, 1],
            robots_work_capacity=[1, 1], max_speed=1, n_agents=1), config))
    payload = dict(format="capam_mrta", version=DATA_VERSION, generation=asdict(config),
                   units=dict(representation="normalized", time="seconds", coordinate_scale=100., workload_scale=100.),
                   obstacle_map=dict(grid=["...", ".#.", "..."]), instances=instances)
    path = Path(directory) / "obstacles.pkl"
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


class NavigationTests(unittest.TestCase):
    def test_all_search_and_residual_steps_use_cached_distances(self):
        raw = instance([[0, 0]], [[0, 0]], [3], speed=2)
        tables = dict(task_to_task=np.zeros((1, 1)), robot_to_task=np.array([[10.]]))
        data = prepare(raw, tables)
        self.assertEqual(validate_solution(prepare(raw), solve(prepare(raw), limit=1))["tasks_done"], 1)
        for backend in ("python", "cpp"):
            if backend == "cpp" and not shutil.which("g++"):
                continue
            for algorithm in ("construct", "ils", "eils"):
                with self.subTest(backend=backend, algorithm=algorithm), \
                        patch("EILS.python_solver.np.hypot", side_effect=AssertionError("Euclidean distance used")):
                    result = solve(data, backend=backend, algorithm=algorithm, limit=3)
                    self.assertEqual(result["finish_times"].tolist(), [6.])
                    self.assertEqual(validate_solution(data, result), dict(tasks_done=0, cost=2.))

    def test_geometry_scale_padding_and_backend_parity(self):
        with tempfile.TemporaryDirectory(dir=str(HERE)) as directory:
            path = obstacle_fixture(directory)
            dataset, obstacle_map, extent = load_evaluation_data(path, offset=1, num_samples=1)
            self.assertEqual(extent, 2.)
            navigation, _ = precompute_navigation(dataset[0], obstacle_map, extent,
                                                   path_planner="visibility_dijkstra")
            data = prepare(dataset[0], navigation)
            self.assertEqual(data["robot_to_task"].shape, (1, 3))
            direct = np.linalg.norm(data["loc"][0] - data["robots_start_location"][0])
            self.assertGreater(data["robot_to_task"][0, 0], direct + .1)
            planner = make_planner(obstacle_map, path_planner="visibility_dijkstra")
            for robot in range(data["m"]):
                for task in range(data["n"]):
                    points = get_path(navigation, data["n"] + robot, task) / extent
                    self.assertTrue(all(planner.segment_is_free(a, b) for a, b in zip(points, points[1:])))
                    self.assertAlmostEqual(np.linalg.norm(np.diff(points * extent, axis=0), axis=1).sum(),
                                           data["robot_to_task"][robot, task], places=6)
            for algorithm in ("construct", "ils", "eils"):
                for service_time in ("continuous", "ceil"):
                    python = solve(data, algorithm=algorithm, service_time=service_time, limit=7)
                    validate_solution(data, python, service_time)
                    if shutil.which("g++"):
                        cpp = solve(data, algorithm=algorithm, service_time=service_time, limit=7, backend="cpp")
                        self.assertEqual(python["routes"], cpp["routes"])
                        np.testing.assert_allclose(python["finish_times"], cpp["finish_times"], atol=1e-10, rtol=1e-12)
                        self.assertEqual(validate_solution(data, python, service_time),
                                         validate_solution(data, cpp, service_time))

    def test_parallel_precomputation_cache_reuse_and_persisted_audit(self):
        with tempfile.TemporaryDirectory(dir=str(HERE)) as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            data = obstacle_fixture(root)
            cache = root / "cache"
            args = ["--data", str(data), "--cache_dir", str(cache), "--offset", "1", "--num_samples", "2",
                    "--algorithms", "eils", "--profiles", "tight", "slack", "--repeats", "2",
                    "--limit", "4", "--progress_every", "0"]
            first = root / "parallel"
            cold = evaluate(args + ["--output_dir", str(first), "--workers", "2", "--path_workers", "2"])
            self.assertEqual(cold["path_planner"], "visibility_dijkstra")
            self.assertEqual(cold["cache_misses"], 2)
            self.assertGreater(cold["path_planning_seconds_total"], 0)
            self.assertEqual(cold["path_scenarios"], 2)  # Shared by eight solves.
            second = root / "serial"
            with patch("z_test_path.visibility_planner.VisibilityGraphPlanner._build_graph",
                       side_effect=AssertionError("Cache hit rebuilt graph")):
                warm = evaluate(args + ["--output_dir", str(second)])
                self.assertEqual(verify(second)["solutions_replayed"], 8)
            self.assertEqual(warm["cache_hits"], 2)
            self.assertEqual(warm["path_planning_seconds_total"], 0.)
            self.assertGreater(warm["cache_read_seconds_total"], 0.)
            self.assertEqual(read_routes(first), read_routes(second))
            a, b = read_rows(first), read_rows(second)
            for key, row in a.items():
                self.assertEqual({k: v for k, v in row.items() if k not in ("cpu_seconds", "inference_seconds")},
                                 {k: v for k, v in b[key].items() if k not in ("cpu_seconds", "inference_seconds")})
            if shutil.which("g++"):
                cpp = root / "cpp"
                evaluate(args + ["--backend", "cpp", "--output_dir", str(cpp)])
                self.assertEqual(verify(cpp)["solutions_replayed"], 8)
                self.assertEqual(compare(cpp, second)["status"], "passed")
            # Audit is self-contained even after the shared navigation cache is gone.
            shutil.rmtree(cache)
            with patch("EILS.navigation.precompute_navigation", side_effect=AssertionError("Audit planned paths")):
                self.assertEqual(verify(first)["solutions_replayed"], 8)
            with (first / "path_distances.npz").open("ab") as handle:
                handle.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "snapshot hash changed"):
                verify(first)

    def test_epsilon_changes_cache_key_and_rebuild_recomputes(self):
        with tempfile.TemporaryDirectory(dir=str(HERE)) as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            path = obstacle_fixture(root)
            args = ["--data", str(path), "--cache_dir", str(root / "cache"), "--num_samples", "1",
                    "--algorithms", "eils", "--profiles", "tight", "--limit", "1", "--progress_every", "0"]
            first = evaluate(args + ["--output_dir", str(root / "first")])
            changed = evaluate(args + ["--visibility_epsilon", "1e-4", "--output_dir", str(root / "changed")])
            self.assertNotEqual(first["navigation_keys"], changed["navigation_keys"])
            rebuilt = evaluate(args + ["--rebuild_cache", "--output_dir", str(root / "rebuilt")])
            self.assertEqual(first["navigation_keys"], rebuilt["navigation_keys"])
            self.assertEqual(rebuilt["cache_misses"], 1)
            self.assertGreater(rebuilt["path_planning_seconds_total"], 0)
            self.assertEqual(read_routes(root / "first"), read_routes(root / "rebuilt"))

    def test_planner_selection_and_invalid_tables(self):
        with tempfile.TemporaryDirectory(dir=str(HERE)) as directory, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            path = obstacle_fixture(directory)
            for args in (["--data", str(path), "--path_planner", "euclidean"], ["--path_workers", "0"],
                         ["--data", str(HERE.parent / "data/mrta_gini"), "--path_planner", "visibility_dijkstra"]):
                with self.assertRaises(SystemExit):
                    evaluate(args + ["--output_dir", str(Path(directory) / "unused")])
        raw = instance([[0, 0]], [[0, 0]], [1])
        for value in (np.nan, -1, np.inf):
            with self.assertRaises(ValueError):
                prepare(raw, dict(task_to_task=[[0]], robot_to_task=[[value]]))


if __name__ == "__main__":
    unittest.main()
