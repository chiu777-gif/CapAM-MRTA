"""Independent insertion oracle, route replay and deadline edge cases."""
import unittest
import numpy as np
import torch

from EILS.solver import prepare, solve, validate_solution
from problems.mrta.data import collate_mrta
from problems.mrta.state_mrta import StateMRTA


def instance(xy, starts, deadlines, workloads=None, capacities=None, speed=1.0):
    n, m = len(xy), len(starts)
    return dict(loc=np.array(xy), robots_start_location=np.array(starts),
                deadline=np.array(deadlines), workload=np.ones(n) if workloads is None else np.array(workloads),
                robots_work_capacity=np.ones(m) if capacities is None else np.array(capacities),
                max_speed=speed, n_agents=m)


def replay(data, route, robot):
    time = 0.0
    location = data["robots_start_location"][robot]
    finishes = []
    for task in route:
        time += np.linalg.norm(data["loc"][task] - location) / data["speed"]
        time += data["workload"][task] / data["robots_work_capacity"][robot]
        location = data["loc"][task]
        finishes.append(time)
    return np.array(finishes)


def brute_construction(data):
    """Recompute entire proposed routes, with no maxshift/candidate cache."""
    routes = [[] for _ in range(data["m"])]
    pending = set(range(data["n"]))
    while pending:
        best = None
        minimum = float("inf")
        for task in sorted(pending):
            for robot, route in enumerate(routes):
                old = replay(data, route, robot)
                for position in range(len(route) + 1):
                    proposed = route[:position] + [task] + route[position:]
                    finished = replay(data, proposed, robot)
                    if (finished > data["deadline"][proposed]).any():
                        continue
                    shift = finished[position+1] - old[position] if position < len(route) else (
                        finished[-1] - (old[-1] if len(old) else 0))
                    if shift < minimum - 1e-12:
                        minimum = shift
                        best = (task, robot, proposed)
        if best is None:
            break
        task, robot, proposed = best
        routes[robot] = proposed
        pending.remove(task)
    return routes


class SolverTests(unittest.TestCase):
    def test_insertion_matches_exhaustive_oracle(self):
        rng = np.random.RandomState(123)
        for _ in range(30):
            data = prepare(instance(rng.rand(9, 2), rng.rand(3, 2), rng.uniform(0.5, 4, 9),
                                    rng.uniform(0.1, 0.5, 9), rng.uniform(0.5, 1.5, 3)))
            expected = brute_construction(data)
            result = solve(data, algorithm="construct")
            self.assertEqual(result["native_completed_tasks"], sum(map(len, expected)))
            for robot, route in enumerate(expected):
                self.assertEqual(result["routes"][robot][:len(route)], route)
            validate_solution(data, result)

    def test_deadline_equality_zero_cost_and_impossible_task(self):
        data = prepare(instance([[0, 0], [0, 0]], [[0, 0]], [1.0, 1.5]))
        result = solve(data, algorithm="construct")
        self.assertEqual(result["routes"], [[0, 1]])
        self.assertEqual(validate_solution(data, result), {"tasks_done": 1, "cost": 2.0 / 1.5})

    def test_no_feasible_task_and_empty_tours(self):
        data = prepare(instance([[10, 10], [20, 20]], [[0, 0], [0, 1], [1, 0]], [0.1, 0.1]))
        for algorithm in ("ils", "eils"):
            result = solve(data, algorithm=algorithm, limit=10)
            self.assertEqual(result["native_completed_tasks"], 0)
            self.assertEqual(validate_solution(data, result)["tasks_done"], 0)

    def test_all_feasible_early_exit(self):
        data = prepare(instance([[0, 0]] * 4, [[0, 0], [0, 0]], [100] * 4))
        for algorithm in ("ils", "eils"):
            result = solve(data, algorithm=algorithm, limit=100)
            self.assertEqual(result["iterations"], 1)
            self.assertEqual(validate_solution(data, result), {"tasks_done": 4, "cost": 0.0})

    def test_continuous_and_original_ceil_modes(self):
        data = prepare(instance([[0, 0]], [[0, 0]], [0.75], [1], [2]))
        continuous = solve(data, algorithm="construct")
        discrete = solve(data, algorithm="construct", service_time="ceil")
        self.assertEqual(validate_solution(data, continuous)["tasks_done"], 1)
        self.assertEqual(validate_solution(data, discrete, "ceil")["tasks_done"], 0)

    def test_unit_rescaling_and_inactive_padding(self):
        raw = instance([[10, 0], [0, 20]], [[0, 0], [900, 900]], [11, 25], [1, 2], [1, 100])
        raw["n_agents"] = 1
        original = solve(prepare(raw), limit=10)
        scaled = dict(raw)
        for key in ("loc", "robots_start_location", "max_speed"):
            scaled[key] = np.asarray(raw[key]) / 100
        for key in ("workload", "robots_work_capacity"):
            scaled[key] = raw[key] / 100
        normalized = solve(prepare(scaled), limit=10)
        np.testing.assert_allclose(original["finish_times"], normalized["finish_times"])
        self.assertTrue((normalized["robots"] == 0).all())

    def test_stochastic_reproducibility_and_full_budget(self):
        rng = np.random.RandomState(2026)
        data = prepare(instance(rng.rand(25, 2), rng.rand(3, 2), rng.uniform(0.5, 2.5, 25),
                                rng.uniform(0.1, 0.5, 25), rng.uniform(0.5, 1.5, 3)))
        for algorithm in ("ils", "eils"):
            for profile in ("tight", "slack"):
                first = solve(data, algorithm=algorithm, profile=profile, limit=50, seed=314)
                second = solve(data, algorithm=algorithm, profile=profile, limit=50, seed=314)
                self.assertEqual(first["routes"], second["routes"])
                validate_solution(data, first)
                self.assertGreaterEqual(first["iterations"], 50)
                if algorithm == "eils":
                    self.assertEqual(first["iterations"], 50)

    def test_single_robot_agrees_with_shared_simulator(self):
        raw = instance([[1, 0], [2, 1], [0, 3]], [[0, 0]], [2, 3, 4], [1, 2, 1], [1])
        raw["depot"] = [[0, 0]]
        tensors = {key: torch.as_tensor(value, dtype=torch.long if key == "n_agents" else torch.float32)
                   for key, value in raw.items()}
        result = solve(prepare(tensors), algorithm="eils", limit=30)
        state = StateMRTA.initialize(collate_mrta([tensors]))
        for task in result["routes"][0]:
            state = state.update(torch.tensor([task + 1]))
        metrics = validate_solution(prepare(tensors), result)
        self.assertAlmostEqual(state.get_final_cost().item(), metrics["cost"], places=5)
        self.assertEqual(state.tasks_done_success.item(), metrics["tasks_done"])

    def test_reject_invalid_data_and_parameters(self):
        with self.assertRaises(ValueError):
            prepare(instance([[0, 0]], [[0, 0]], [0]))
        data = prepare(instance([[0, 0]], [[0, 0]], [1]))
        for kwargs in ({"limit": 0}, {"coefficient": 0}, {"coefficient": 1.1}, {"seed": -1}):
            with self.assertRaises(ValueError):
                solve(data, **kwargs)


if __name__ == "__main__":
    unittest.main()
