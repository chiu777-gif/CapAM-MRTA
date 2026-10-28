"""Full-coverage cost selection, legacy behavior and both backend/CLI paths."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from EILS.evaluate import main as evaluate
from EILS.python_solver import Search
from EILS.solver import prepare, solve, validate_solution
from EILS import test_parallel_evaluation
from EILS.test_solver import instance
from EILS.verify_results import verify


def objective_fixture():
    return prepare(instance(
        [[0, 1], [2, 2], [1, 3], [4, 1], [4, 2], [2, 1], [2, 4], [2, 4]],
        [[2, 3], [3, 1]], [3, 14, 13, 9, 7, 6, 14, 15],
        [3, 3, 2, 1, 2, 2, 2, 1], [1, 2]))


class ObjectiveTests(unittest.TestCase):
    def test_cost_selection_changes_solution_and_default_is_unchanged(self):
        data = objective_fixture()
        for algorithm in ("ils", "eils"):
            options = dict(algorithm=algorithm, limit=30, seed=19)
            default = solve(data, **options)
            completion = solve(data, objective="completion", **options)
            cost = solve(data, objective="cost", **options)
            self.assertEqual(default["routes"], completion["routes"])
            a, b = validate_solution(data, completion), validate_solution(data, cost)
            self.assertEqual(a["tasks_done"], 7)
            self.assertAlmostEqual(a["cost"], 4.673254945945694)
            self.assertAlmostEqual(b["cost"], 3.5351837584879964)
            self.assertLess(b["cost"], a["cost"])
            self.assertNotEqual(completion["routes"], cost["routes"])
        with self.assertRaisesRegex(ValueError, "objective"):
            solve(data, objective="unknown")

    def test_scoring_includes_residual_and_restores_search_state(self):
        data = objective_fixture()
        search = Search(data, "continuous")
        search.construct()
        tours = [tour.copy() for tour in search.tours]
        used = search.used.copy()
        value = search.completed_cost()
        self.assertGreater(value, 0.)  # Feasible prefix alone would always cost zero.
        self.assertEqual(tours, search.tours)
        np.testing.assert_array_equal(used, search.used)
        self.assertEqual(value, search.completed_cost())
        completed = search.complete_all()
        late = completed["finish_times"] > data["deadline"]
        self.assertAlmostEqual(value, (completed["finish_times"][late] / data["deadline"][late]).sum())

    def test_zero_cost_is_a_global_lower_bound_and_exits(self):
        data = prepare(instance([[0, 0]] * 3, [[0, 0]], [10] * 3))
        for algorithm in ("ils", "eils"):
            result = solve(data, algorithm=algorithm, objective="cost", limit=10)
            self.assertEqual(result["iterations"], 1)
            self.assertEqual(validate_solution(data, result)["cost"], 0.)

    @unittest.skipUnless(shutil.which("g++"), "C++ parity requires g++")
    def test_cost_python_cpp_parity_with_and_without_navigation(self):
        base = objective_fixture()
        tables = dict(base,
                      task_to_task=np.hypot(base["loc"][:, None, 0] - base["loc"][None, :, 0],
                                            base["loc"][:, None, 1] - base["loc"][None, :, 1]),
                      robot_to_task=np.hypot(base["robots_start_location"][:, None, 0] - base["loc"][None, :, 0],
                                             base["robots_start_location"][:, None, 1] - base["loc"][None, :, 1]))
        for data in (base, tables):
            for algorithm in ("ils", "eils"):
                for profile in ("tight", "slack"):
                    for mode in ("continuous", "ceil"):
                        options = dict(algorithm=algorithm, profile=profile, service_time=mode,
                                       limit=30, seed=19, objective="cost")
                        python = solve(data, backend="python", **options)
                        cpp = solve(data, backend="cpp", **options)
                        self.assertEqual(python["routes"], cpp["routes"])
                        for field in ("iterations", "best_iteration", "native_completed_tasks"):
                            self.assertEqual(python[field], cpp[field])
                        np.testing.assert_allclose(python["finish_times"], cpp["finish_times"], atol=1e-10)
                        validate_solution(data, python, mode)
                        validate_solution(data, cpp, mode)

    def test_cost_parallel_matches_serial_and_audits_objective(self):
        here = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(dir=str(here)) as directory:
            # Reuse the integration assertion helper, not its full test suite.
            helper = test_parallel_evaluation.ParallelEvaluationTests()
            helper.compare_runs(directory, ["--backend", "python", "--num_samples", "2",
                                           "--objective", "cost", "--limit", "4"], 8)
            output = Path(directory) / "parallel"
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["objective"], "cost")
            self.assertIn("最小化", (output / "REPORT.md").read_text())
            summary["objective"] = "completion"
            (output / "summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, "objective"), contextlib.redirect_stdout(io.StringIO()):
                verify(output)


if __name__ == "__main__":
    unittest.main()
