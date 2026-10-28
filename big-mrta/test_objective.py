"""Verify the selected penalty, assignment and full-visit semantics."""
import unittest

from baseline import choose_cost, prepare, replay_full_visit, solve_full_visit, solve_upstream
from test_baseline import physical_instance
import numpy as np
import torch


class ObjectiveTests(unittest.TestCase):
    def test_minimum_sum_matching_and_unmatched_current_robot(self):
        state = np.zeros((2, 9))
        times = {'robot-1': np.array([[0., 8., 2.]] * 3),
                 'robot-2': np.array([[0., 3., 5.]] * 3)}
        deadlines = np.array([0., 1., 1.])
        # Cross assignment costs 2+3; diagonal assignment costs 8+5.
        self.assertEqual(choose_cost(0, state, [1, 2], times, deadlines), 2)
        self.assertEqual(choose_cost(1, state, [1, 2], times, deadlines), 1)
        # One task is initially matched to robot 2; robot 1 still must progress.
        self.assertEqual(choose_cost(0, state, [1], times, deadlines), 1)

    def test_deadline_equality_availability_and_task_indices(self):
        state = np.zeros((1, 9))
        times = {'robot-1': np.array([[0., 2., 1.]] * 3)}
        deadlines = np.array([0., 1., 1.])
        self.assertEqual(choose_cost(0, state, [1, 2], times, deadlines), 2)
        state[0, 1] = 2.
        self.assertEqual(choose_cost(0, state, [2, 1], times, deadlines), 2)

    def test_all_tasks_are_visited_and_replay_under_both_objectives(self):
        torch.set_num_threads(1)
        scenario = prepare(physical_instance([[0., 0.], [0., 0.]], [[0., 0.]],
                                             [.1, 1.], workload=[2., 3.]))
        legacy = solve_full_visit(scenario)
        completion = solve_full_visit(scenario, objective='completion')
        self.assertEqual(legacy['events'], completion['events'])
        cost = solve_full_visit(scenario, objective='cost')
        self.assertEqual(completion['events'][0]['task'], 1)
        self.assertEqual(cost['events'][0]['task'], 2)
        for result in (completion, cost):
            self.assertEqual(sorted(event['task'] for event in result['events']), [1, 2])
            replay_full_visit(scenario, result)
        self.assertEqual(cost['relaxed_decisions'], 0)
        # A lower immediate penalty can increase final schedule cost. This
        # deliberately prevents tests/docs from implying a global guarantee.
        self.assertGreater(replay_full_visit(scenario, cost)[0], replay_full_visit(scenario, completion)[0])
        with self.assertRaises(ValueError):
            solve_upstream(scenario, objective='cost')
        with self.assertRaises(ValueError):
            solve_full_visit(scenario, objective='invalid')


if __name__ == '__main__':
    unittest.main()
