"""Check obstacle exclusion, coordinate conventions, and dataset compatibility."""
import contextlib
import io
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch

from problems.mrta.config import MRTAConfig
from problems.mrta.data import RandomMRTADataset, collate_mrta, load_file_dataset
from z_test_path.generate_test_set import ObstacleMap, ObstacleMRTADataset, main


class ObstacleGenerationTests(unittest.TestCase):
    def assert_free(self, points, obstacle_map, extent=1.):
        for x, y in points.double().tolist():
            self.assertTrue(0 < x < extent and 0 < y < extent)
            # Independent geometric check against every CLOSED obstacle cell.
            for row, line in enumerate(obstacle_map.grid):
                for column, symbol in enumerate(line):
                    if symbol == '#':
                        blocked = (column * extent / obstacle_map.width <= x <=
                                   (column + 1) * extent / obstacle_map.width and
                                   (obstacle_map.height - row - 1) * extent / obstacle_map.height <= y <=
                                   (obstacle_map.height - row) * extent / obstacle_map.height)
                        self.assertFalse(blocked, (x, y, row, column))

    def test_invalid_maps(self):
        for grid in [(), ('',), ('..', '.'), ('..', ''), (' .', '..'), ('x.', '..'), ('##', '##')]:
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                ObstacleMap(grid)

    def test_all_positions_avoid_obstacles_including_inactive_robots(self):
        obstacle_map = ObstacleMap(('.##.', '#...', '..##'))
        dataset = ObstacleMRTADataset(MRTAConfig(graph_size=50, robot_counts=(1, 4)), 30, 1234, obstacle_map)
        for item in dataset:
            for field in ('loc', 'robots_start_location', 'depot'):
                self.assert_free(item[field], obstacle_map)
            self.assertEqual(item['robots_start_location'].shape, (4, 2))

    def test_single_free_cell_orientation_and_continuous_sampling(self):
        for grid, x_range, y_range in [
                (('.#', '##'), (0., .5), (.5, 1.)),
                (('##', '#.'), (.5, 1.), (0., .5))]:
            with self.subTest(grid=grid):
                item = ObstacleMRTADataset(MRTAConfig(graph_size=200), 1, 7, ObstacleMap(grid))[0]
                points = torch.cat([item[key] for key in ('loc', 'robots_start_location', 'depot')])
                self.assertTrue(((points[:, 0] > x_range[0]) & (points[:, 0] < x_range[1])).all())
                self.assertTrue(((points[:, 1] > y_range[0]) & (points[:, 1] < y_range[1])).all())
                self.assertGreater(points[:, 0].unique().numel(), 100)
                self.assertGreater(points[:, 1].unique().numel(), 100)

    def test_seed_repeatability_random_depot_and_original_attribute_distributions(self):
        config = MRTAConfig(graph_size=5, robot_counts=(2, 4))
        obstacle_map = ObstacleMap(('...', '.#.'))
        first = ObstacleMRTADataset(config, 2, 1234, obstacle_map)
        same = ObstacleMRTADataset(config, 2, 1234, obstacle_map)
        other = ObstacleMRTADataset(config, 2, 4321, obstacle_map)
        base = RandomMRTADataset(config, 2, 1234)
        for index in range(2):
            for field in first[index]:
                torch.testing.assert_close(first[index][field], same[index][field], rtol=0, atol=0)
            for field in ('n_agents', 'max_speed', 'deadline', 'workload', 'robots_work_capacity'):
                torch.testing.assert_close(first[index][field], base[index][field], rtol=0, atol=0)
        self.assertFalse(torch.equal(first[0]['depot'], first[1]['depot']))
        self.assertFalse(torch.equal(first[0]['loc'], other[0]['loc']))

    def test_normalization_with_rectangular_map(self):
        obstacle_map = ObstacleMap(('##..', '.###'))
        config = MRTAConfig(graph_size=20, map_size=200., coordinate_scale=50.)
        item = ObstacleMRTADataset(config, 1, 1234, obstacle_map)[0]
        for field in ('loc', 'robots_start_location', 'depot'):
            self.assert_free(item[field], obstacle_map, extent=4.)
        torch.testing.assert_close(item['max_speed'], torch.tensor(.02))

    def test_cli_roundtrip_metadata_overwrite_and_external_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            map_path = root / 'map.txt'
            # Accept UTF-8 BOM and normal Windows line endings.
            map_path.write_bytes(b'\xef\xbb\xbf.##\r\n...\r\n')
            output = root / 'samples.pkl'
            args = ['--map_file', str(map_path), '--graph_size', '8', '--dataset_size', '3',
                    '--n_agents', '2', '--seed', '1234', '--filename', str(output)]
            command = [sys.executable, str(REPO / 'z_test_path/generate_test_set.py')] + args
            generated = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=30)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            data = load_file_dataset(output)
            self.assertEqual(len(data), 3)
            self.assertEqual(collate_mrta([data[0], data[1]])['loc'].shape, (2, 8, 2))
            with output.open('rb') as handle:
                payload = pickle.load(handle)
            metadata = json.loads(output.with_suffix('.json').read_text())
            self.assertEqual(payload['obstacle_map']['grid'], ['.##', '...'])
            self.assertEqual(payload['obstacle_map'], metadata['obstacle_map'])
            self.assertFalse(payload['path_distances_precomputed'])
            for item in data:
                for field in ('loc', 'robots_start_location', 'depot'):
                    self.assert_free(item[field], ObstacleMap(('.##', '...')))
            original = output.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main(args)
                self.assertEqual(original, output.read_bytes())
                main(args + ['--overwrite'])
            regenerated = load_file_dataset(output)
            for index in range(len(data)):
                for field in data[index]:
                    torch.testing.assert_close(data[index][field], regenerated[index][field], rtol=0, atol=0)

    def test_multi_size_and_invalid_inputs_create_no_partial_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            map_path = root / 'map.txt'
            map_path.write_text('..\n.#\n')
            common = ['--map_file', str(map_path), '--output_dir', str(root / 'data')]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                for extra in [
                        ['--graph_sizes', '2', '-1'], ['--graph_sizes', '2', '2'],
                        ['--dataset_size', '0'], ['--seed', '-1'],
                        ['--filename', str(root / 'wrong.txt')]]:
                    with self.subTest(extra=extra), self.assertRaises(SystemExit):
                        main(common + extra)
                    self.assertFalse((root / 'data').exists())
                outputs = main(common + ['--graph_sizes', '2', '4', '--dataset_size', '1'])
            self.assertEqual([load_file_dataset(path)[0]['loc'].shape[0] for path in outputs], [2, 4])


if __name__ == '__main__':
    unittest.main()
