"""Check metric weighting, batch allocation, and CPU/GPU identification."""
import unittest
from unittest.mock import patch

import torch

from z_test_path.inference import synchronize
from z_test_path.reporting import device_metadata, summarize_rows


class ReportingTests(unittest.TestCase):
    def rows(self):
        return [
            {'task_count': 2, 'tasks_done': 1, 'completion_rate': .5, 'cost': 10.,
             'batch_id': 0, 'batch_scenarios': 2, 'inference_seconds': 2.,
             'path_planning_seconds': 0., 'cache_hit': True},
            {'task_count': 8, 'tasks_done': 8, 'completion_rate': 1., 'cost': 20.,
             'batch_id': 0, 'batch_scenarios': 2, 'inference_seconds': 2.,
             'path_planning_seconds': 0., 'cache_hit': True},
            {'task_count': 1, 'tasks_done': 0, 'completion_rate': 0., 'cost': 30.,
             'batch_id': 1, 'batch_scenarios': 1, 'inference_seconds': 3.,
             'path_planning_seconds': 4., 'cache_hit': False},
        ]

    def test_weighted_completion_and_amortized_group_times(self):
        rows = self.rows()
        metadata = {'device_type': 'CPU', 'device': 'cpu'}
        overall = summarize_rows(rows, metadata)
        first = summarize_rows([rows[0], rows[2]], metadata)
        second = summarize_rows([rows[1]], metadata)
        self.assertAlmostEqual(overall['completion_rate'], 9 / 11)
        self.assertEqual(overall['mean_scenario_completion_rate'], .5)
        self.assertEqual(overall['mean_cost'], 20.)
        self.assertEqual(overall['inference_seconds_total'], 7.)
        self.assertAlmostEqual(overall['inference_seconds_per_scenario'], 7 / 3)
        self.assertEqual(overall['path_planning_seconds_total'], 4.)
        self.assertEqual(first['inference_seconds_total'], 5.)
        self.assertEqual(second['inference_seconds_total'], 2.)
        self.assertAlmostEqual(first['completion_rate'], 1 / 3)
        self.assertEqual(first['mean_scenario_completion_rate'], .25)
        self.assertEqual(overall['cache_hits'], 2)
        self.assertEqual(overall['cache_misses'], 1)

    def test_gpu_metadata_and_device_synchronization(self):
        device = torch.device('cuda:2')
        with patch('torch.cuda.get_device_name', return_value='Test GPU'):
            metadata = device_metadata(device, 'cuda:2', 1)
        self.assertEqual(metadata['device_type'], 'GPU')
        self.assertEqual(metadata['device_name'], 'Test GPU')
        self.assertEqual(metadata['device'], 'cuda:2')
        self.assertEqual(metadata['path_device'], 'CPU')
        with patch('torch.cuda.synchronize') as sync:
            synchronize(device)
            sync.assert_called_once_with(device)
        with patch('torch.cuda.synchronize') as sync:
            synchronize(torch.device('cpu'))
            sync.assert_not_called()


if __name__ == '__main__':
    unittest.main()
