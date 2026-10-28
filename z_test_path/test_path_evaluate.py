"""Geometry, checkpoint compatibility, table simulation and evaluator checks."""
import contextlib
import csv
import io
import json
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import torch

from nets.attention_model import AttentionModel
from problems.mrta.config import DATA_VERSION, MRTAConfig
from problems.mrta.data import collate_mrta, validate_instance
from problems.mrta.problem_mrta import MRTA
from z_test_path.cankao_planner import GridPathPlanner
from z_test_path.evaluate import _plan_case, main as evaluate
from z_test_path.generate_test_set import ObstacleMap, ObstacleMRTADataset, main as generate
from z_test_path.path_adapter import PathMRTA, adapt_model
from z_test_path.path_planner import get_path, make_planner, precompute_navigation, prepare_path_dataset
from z_test_path.plot_test_set import load_routes
from z_test_path.reporting import TIMING_FIELDS


def scenario(loc, robots, agents=None):
    return validate_instance({
        'loc': loc, 'robots_start_location': robots, 'n_agents': len(robots) if agents is None else agents,
        'depot': [[.1, .1]], 'max_speed': 1., 'deadline': [10.] * len(loc),
        'workload': [0.] * len(loc), 'robots_work_capacity': [1.] * len(robots),
    })


class PathEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_reference_paths_boundary_collision_and_corner_cutting(self):
        obstacle_map = ObstacleMap(('...', '.#.', '...'))
        item = scenario([[.15, .5], [.85, .5]], [[.15, .8]])
        navigation, _ = precompute_navigation(item, obstacle_map, 1., subdivisions=3)
        planner = make_planner(obstacle_map, 3)
        expected, distance = planner.plan(item['loc'][0].numpy(), item['loc'][1].numpy())
        np.testing.assert_array_equal(get_path(navigation, 0, 1), expected)
        self.assertGreater(distance, .7)
        self.assertAlmostEqual(float(navigation['task_to_task'][0, 1]), distance, places=6)
        np.testing.assert_array_equal(get_path(navigation, 1, 0), expected[::-1])
        for route in navigation['paths'].values():
            self.assertTrue(all(planner.segment_is_free(a, b) for a, b in zip(route[:-1], route[1:])))
        with self.assertRaises(ValueError):
            planner.plan([1 / 3, .5], [.1, .1])  # On the closed obstacle boundary.
        diagonal = make_planner(ObstacleMap(('.#', '#.')), 2)
        with self.assertRaises(ValueError):
            diagonal.plan([.25, .75], [.75, .25])

    def test_unreachable_scene_is_reported_and_inactive_robots_are_ignored(self):
        obstacle_map = ObstacleMap(('...', '###', '...'))
        item = scenario([[.2, .85], [.8, .85]], [[.5, .85], [.5, .15]], agents=1)
        navigation, _ = precompute_navigation(item, obstacle_map, 1., subdivisions=2)
        np.testing.assert_array_equal(navigation['robot_to_task'][1], 0.)
        item['n_agents'] = torch.tensor(2)
        with self.assertRaisesRegex(ValueError, 'Scenario index 17'):
            _plan_case(17, item, obstacle_map, 1., 2, None, False)

    def test_empty_map_zero_length_paths_and_unit_scaling(self):
        obstacle_map = ObstacleMap(('..', '..'))
        item = scenario([[.1, .2], [.1, .2], [.8, .9]], [[.4, .5]])
        navigation, _ = precompute_navigation(item, obstacle_map, 1., subdivisions=2)
        expected = torch.cdist(item['loc'][None], item['loc'][None])[0].numpy()
        np.testing.assert_allclose(navigation['task_to_task'], expected, atol=1e-7)
        self.assertEqual(navigation['task_to_task'][0, 1], 0.)
        self.assertEqual(get_path(navigation, 0, 0).shape, (2, 2))
        larger = dict(item)
        larger['loc'], larger['robots_start_location'] = item['loc'] * 4, item['robots_start_location'] * 4
        scaled, _ = precompute_navigation(larger, obstacle_map, 4., subdivisions=2)
        np.testing.assert_allclose(scaled['task_to_task'], navigation['task_to_task'] * 4)
        np.testing.assert_allclose(scaled['robot_to_task'], navigation['robot_to_task'] * 4)

    def test_cache_reuse_and_geometry_invalidation(self):
        item = scenario([[.1, .2], [.8, .9]], [[.4, .5]])
        obstacle_map = ObstacleMap(('..', '..'))
        with tempfile.TemporaryDirectory() as directory:
            first, stats = precompute_navigation(item, obstacle_map, 1., 2, directory)
            self.assertFalse(stats['cache_hit'])
            with patch.object(GridPathPlanner, 'plan', side_effect=AssertionError('Cache hit ran a search')):
                same, stats = precompute_navigation(item, obstacle_map, 1., 2, directory)
            self.assertTrue(stats['cache_hit'])
            self.assertEqual(first['key'], same['key'])
            changed = dict(item)
            changed['loc'] = item['loc'].clone()
            changed['loc'][0, 0] += .01
            second, stats = precompute_navigation(changed, obstacle_map, 1., 2, directory)
            self.assertFalse(stats['cache_hit'])
            self.assertNotEqual(first['key'], second['key'])
            refined, _ = precompute_navigation(item, obstacle_map, 1., 3, directory)
            self.assertNotEqual(first['key'], refined['key'])

    def test_state_uses_cached_start_and_task_distances_at_deadline_boundary(self):
        item = scenario([[.1, .1], [.2, .1]], [[0., .1]])
        item['deadline'] = torch.tensor([2., 8.])
        navigation = {'task_to_task': np.array([[0., 5.], [5., 0.]], dtype=np.float32),
                      'robot_to_task': np.array([[3., 4.]], dtype=np.float32)}
        dataset = prepare_path_dataset([item], [0], [navigation], 'synthetic')
        batch = collate_mrta([dataset[0]])
        initial = PathMRTA.make_state(batch)
        first = initial.update(torch.tensor([1]))
        final = first.update(torch.tensor([2]))
        self.assertEqual(type(final).__name__, 'PathState')
        torch.testing.assert_close(final.tasks_finish_time, torch.tensor([[3., 8.]]))
        self.assertEqual(final.tasks_done_success.item(), 1)
        self.assertEqual(final.get_final_cost().item(), 1.5)
        self.assertFalse(initial.visited.any())
        self.assertEqual(initial.lengths.item(), 0.)

    def test_no_obstacle_checkpoint_behavior_and_unchanged_parameters(self):
        obstacle_map = ObstacleMap(('..', '..'))
        config = MRTAConfig(graph_size=6, robot_counts=(2,), deadline_min=.5, deadline_max=.5)
        items = [ObstacleMRTADataset(config, 2, 1234, obstacle_map)[index] for index in range(2)]
        navigations = [precompute_navigation(item, obstacle_map, 1., 2)[0] for item in items]
        data = prepare_path_dataset(items, [0, 1], navigations, 'empty-map')
        batch = collate_mrta([data[0], data[1]])
        torch.manual_seed(17)
        model = AttentionModel(32, 32, MRTA).eval()
        model.set_decode_type('greedy')
        weights = {key: value.clone() for key, value in model.state_dict().items()}
        with torch.no_grad():
            before_embeddings = model.embedder(batch)[0]
            before_cost, _, before_sequence = model(batch, return_pi=True)
            before_finish = MRTA.replay(batch, before_sequence).tasks_finish_time
            adapt_model(model)
            after_embeddings = model.embedder(batch)[0]
            after_cost, _, after_sequence = model(batch, return_pi=True)
            after_finish = PathMRTA.replay(batch, after_sequence).tasks_finish_time
        torch.testing.assert_close(before_embeddings, after_embeddings, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(before_sequence, after_sequence)
        torch.testing.assert_close(before_cost, after_cost)
        torch.testing.assert_close(before_finish, after_finish)
        self.assertEqual(weights.keys(), model.state_dict().keys())
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, weights[key], rtol=0, atol=0)

    def make_inputs(self, root, encoder='gcapcn'):
        map_path = root / 'map.txt'
        map_path.write_text('..#.\n....\n')
        path = root / 'data.pkl'
        with contextlib.redirect_stdout(io.StringIO()):
            generate(['--map_file', str(map_path), '--graph_size', '5', '--dataset_size', '3',
                      '--n_agents', '2', '--filename', str(path)])
        model_dir = root / 'model'
        model_dir.mkdir()
        torch.manual_seed(17)
        layers = 1 if encoder == 'gcapcn' else 2
        model = AttentionModel(32, 32, MRTA, encoder=encoder, n_encode_layers=layers).eval()
        torch.save({'model': model.state_dict()}, model_dir / 'epoch-0.pt')
        (model_dir / 'args.json').write_text(json.dumps({
            'problem': 'mrta', 'embedding_dim': 32, 'normalization': 'batch',
            'tanh_clipping': 10., 'schema_version': DATA_VERSION, 'network_version': DATA_VERSION,
            'coordinate_scale': 100., 'workload_scale': 100.,
            'encoder': encoder, 'n_encode_layers': layers,
        }))
        return path, model_dir

    def test_am_encoder_checkpoint_and_path_simulation(self):
        self.check_additional_encoder('am')

    def test_mamba_encoder_checkpoint_and_path_simulation(self):
        self.check_additional_encoder('mamba')

    def check_additional_encoder(self, encoder):
        obstacle_map = ObstacleMap(('..', '..'))
        config = MRTAConfig(graph_size=5, robot_counts=(2,), deadline_min=.5, deadline_max=.5)
        item = ObstacleMRTADataset(config, 1, 1234, obstacle_map)[0]
        navigation, _ = precompute_navigation(item, obstacle_map, 1., 2)
        dataset = prepare_path_dataset([item], [0], [navigation], 'no-obstacles')
        batch = collate_mrta([dataset[0]])
        torch.manual_seed(17)
        model = AttentionModel(32, 32, MRTA, encoder=encoder, n_encode_layers=2).eval()
        model.set_decode_type('greedy')
        original_encoder = model.embedder
        weights = {key: value.clone() for key, value in model.state_dict().items()}
        with torch.no_grad():
            before = model(batch, return_pi=True)
            adapt_model(model)
            after = model(batch, return_pi=True)
            self.assertIs(model.embedder, original_encoder)
            for original, adapted in zip(before, after):
                torch.testing.assert_close(original, adapted)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
            model.set_decode_type('sampling')
            with patch.object(GridPathPlanner, 'plan', side_effect=AssertionError('Inference ran a search')):
                sequences, costs, done = model.sample_many(batch, batch_rep=3)
                replay = PathMRTA.replay(batch, sequences)
            torch.testing.assert_close(costs, replay.get_final_cost()[:, 0])
            self.assertEqual(torch.as_tensor(done).reshape(-1).tolist(),
                             replay.tasks_done_success[:, 0].tolist())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, model_dir = self.make_inputs(root, encoder)
            common = ['--model', str(model_dir), '--data', str(path), '--cache_dir', str(root / 'cache'),
                      '--device', 'cpu', '--path_subdivisions', '2', '--eval_batch_size', '2',
                      '--max_calc_batch_size', '4', '--no_progress_bar']
            with contextlib.redirect_stdout(io.StringIO()):
                for strategy, width in [('greedy', 1), ('sample', 3)]:
                    output = root / strategy
                    result = evaluate(common + ['--output_dir', str(output), '--decode_strategy', strategy,
                                                '--width', str(width)])
                    self.assertEqual(result['encoder'], encoder)
                    self.assertEqual(result['total_tasks'], 15)
                    self.assertIsNone(result['graph_distance'])
                    self.assertFalse(result['encoder_uses_path_distances'])
                    self.assertEqual(len(load_routes(output / 'routes.json', path, 0)), 5)

    def test_sampling_uses_only_tables_and_robot_padding_does_not_change_greedy(self):
        obstacle_map = ObstacleMap(('..', '..'))
        items = [scenario([[.2, .2], [.3, .8], [.8, .7]], [[.1, .1], [.9, .9]]),
                 scenario([[.2, .2], [.3, .8], [.8, .7]],
                          [[.1, .1], [.9, .9], [.1, .9], [.9, .1]])]
        navigations = [precompute_navigation(item, obstacle_map, 1., 2)[0] for item in items]
        padded = prepare_path_dataset(items, [0, 1], navigations, 'padded')
        single = prepare_path_dataset(items[:1], [0], navigations[:1], 'single')
        torch.manual_seed(17)
        model = adapt_model(AttentionModel(32, 32, MRTA).eval())
        model.set_decode_type('greedy')
        batch = collate_mrta([padded[0], padded[1]])
        with patch.object(GridPathPlanner, 'plan', side_effect=AssertionError('Inference ran a search')):
            with torch.no_grad():
                padded_cost, _, padded_sequence = model(batch, return_pi=True)
                single_cost, _, single_sequence = model(collate_mrta([single[0]]), return_pi=True)
                torch.testing.assert_close(padded_cost[:1], single_cost)
                torch.testing.assert_close(padded_sequence[:1], single_sequence)
                model.set_decode_type('sampling')
                sequences, costs, done = model.sample_many(batch, batch_rep=3)
                replay = PathMRTA.replay(batch, sequences)
                torch.testing.assert_close(costs, replay.get_final_cost()[:, 0])
                self.assertEqual(torch.as_tensor(done).reshape(-1).tolist(),
                                 replay.tasks_done_success[:, 0].tolist())

    def test_end_to_end_greedy_sampling_warmup_and_route_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, model_dir = self.make_inputs(root)
            common = ['--model', str(model_dir), '--data', str(path), '--cache_dir', str(root / 'cache'),
                      '--device', 'cpu', '--path_subdivisions', '2', '--eval_batch_size', '2',
                      '--max_calc_batch_size', '4', '--no_progress_bar']
            with contextlib.redirect_stdout(io.StringIO()):
                greedy = evaluate(common + ['--output_dir', str(root / 'greedy')])
                samples = []
                for warmup in (0, 2):
                    output = root / ('sample' + str(warmup))
                    result = evaluate(common + ['--output_dir', str(output), '--decode_strategy', 'sample',
                                                '--width', '3', '--warmup_batches', str(warmup)])
                    self.assertEqual(result['cache_hits'], 3)
                    samples.append(json.loads((output / 'routes.json').read_text())['scenarios'])
            self.assertEqual(greedy['total_tasks'], 15)
            self.assertEqual(greedy['cache_misses'], 3)
            self.assertEqual(samples[0], samples[1])
            self.assertEqual(len(load_routes(root / 'greedy/routes.json', path, 0)), 5)
            original = path.read_bytes()
            path.write_bytes(original + b'changed')
            with self.assertRaisesRegex(ValueError, 'different dataset'):
                load_routes(root / 'greedy/routes.json', path, 0)

    def test_cli_outside_repository_with_parallel_preprocessing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, model_dir = self.make_inputs(root)
            for planner in ('grid_astar', 'visibility_dijkstra'):
                with self.subTest(planner=planner):
                    output = root / planner
                    result = subprocess.run([
                        sys.executable, str(REPO / 'z_test_path/evaluate.py'), '--model', str(model_dir),
                        '--data', str(path), '--output_dir', str(output), '--cache_dir', str(root / 'cache'),
                        '--num_samples', '2', '--device', 'cpu', '--path_workers', '2',
                        '--path_planner', planner, '--path_subdivisions', '2', '--no_progress_bar',
                    ], cwd=root, capture_output=True, text=True, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    summary = json.loads((output / 'summary.json').read_text())
                    self.assertEqual(summary['scenarios'], 2)
                    self.assertEqual(summary['path_planner'], planner)

    def test_visibility_dijkstra_encoders_sampling_and_export_metadata(self):
        for encoder in ('gcapcn', 'am', 'mamba'):
            with self.subTest(encoder=encoder), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                path, model_dir = self.make_inputs(root, encoder)
                original_data = path.read_bytes()
                common = ['--model', str(model_dir), '--data', str(path), '--cache_dir', str(root / 'cache'),
                          '--path_planner', 'visibility_dijkstra', '--visibility_epsilon', '2e-6',
                          '--device', 'cpu', '--eval_batch_size', '2', '--max_calc_batch_size', '4',
                          '--no_progress_bar']
                with contextlib.redirect_stdout(io.StringIO()):
                    for strategy, width in [('greedy', 1), ('sample', 3)]:
                        output = root / strategy
                        summary = evaluate(common + ['--output_dir', str(output), '--decode_strategy', strategy,
                                                     '--width', str(width)])
                        self.assertEqual(summary['path_planner'], 'visibility_dijkstra')
                        self.assertIn('Dijkstra', summary['planner'])
                        self.assertEqual(summary['visibility_epsilon'], 2e-6)
                        self.assertIsNone(summary['path_subdivisions'])
                        self.assertEqual(summary['total_tasks'], 15)
                        self.assertEqual(summary['cache_hits'], 3 if strategy == 'sample' else 0)
                        self.assertEqual(len(load_routes(output / 'routes.json', path, 0)), 5)
                        routes = json.loads((output / 'routes.json').read_text())
                        self.assertEqual(routes['path_planner'], 'visibility_dijkstra')
                        self.assertEqual(routes['visibility_epsilon'], 2e-6)
                        self.assertIsNone(routes['path_subdivisions'])
                        for filename in ('overall.csv', 'scenarios.csv', 'groups.csv', 'batches.csv'):
                            with (output / filename).open(newline='') as handle:
                                for row in csv.DictReader(handle):
                                    self.assertEqual(row['path_planner'], 'visibility_dijkstra')
                                    self.assertEqual(float(row['visibility_epsilon']), 2e-6)
                        report = (output / 'REPORT.md').read_text()
                        self.assertIn('visibility_dijkstra', report)
                        self.assertIn('Dijkstra', report)
                        self.assertIn('2e-06', report)
                self.assertEqual(path.read_bytes(), original_data)

    def test_concise_reports_uneven_batches_and_robot_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, model_dir = self.make_inputs(root)
            with path.open('rb') as handle:
                payload = pickle.load(handle)
            payload['instances'][1]['n_agents'] = torch.tensor(1)
            payload['generation']['robot_counts'] = (1, 2)
            with path.open('wb') as handle:
                pickle.dump(payload, handle)
            output = root / 'reports'
            with contextlib.redirect_stdout(io.StringIO()):
                summary = evaluate([
                    '--model', str(model_dir), '--data', str(path), '--cache_dir', str(root / 'cache'),
                    '--output_dir', str(output), '--device', 'cpu', '--path_subdivisions', '2',
                    '--eval_batch_size', '2', '--no_progress_bar',
                ])
            def read_csv(name):
                with (output / name).open(newline='') as handle:
                    return list(csv.DictReader(handle))

            rows, batches, groups, overall = map(read_csv, ('scenarios.csv', 'batches.csv', 'groups.csv', 'overall.csv'))
            self.assertEqual(len(overall), 1)
            self.assertEqual([int(row['scenarios']) for row in batches], [2, 1])
            self.assertEqual(summary['device_type'], 'CPU')
            self.assertEqual(summary['inference_timing_mode'], 'batch_amortized')
            self.assertEqual(set(summary['by_n_agents']), {'1', '2'})
            self.assertAlmostEqual(summary['inference_seconds_total'],
                                   sum(float(row['inference_seconds']) for row in batches))
            self.assertAlmostEqual(summary['inference_seconds_total'],
                                   sum(float(row['inference_seconds']) for row in rows))
            self.assertAlmostEqual(summary['inference_seconds_total'],
                                   sum(float(row['inference_seconds_total']) for row in groups))
            self.assertAlmostEqual(summary['inference_seconds_per_scenario'],
                                   summary['inference_seconds_total'] / 3)
            def time_fields(row):
                return {key for key in row if '_seconds' in key or '_ms_' in key}

            self.assertEqual(time_fields(summary), set(TIMING_FIELDS))
            self.assertEqual(time_fields(overall[0]), set(TIMING_FIELDS))
            self.assertEqual(set(summary['timing_definitions']), set(TIMING_FIELDS))
            self.assertEqual(time_fields(batches[0]), {'inference_seconds'})
            self.assertEqual(time_fields(rows[0]), {'inference_seconds', 'path_planning_seconds'})
            for group in groups:
                selected = [row for row in rows if row['n_agents'] == group['n_agents']]
                self.assertEqual(group['device_type'], 'CPU')
                self.assertAlmostEqual(float(group['mean_cost']),
                                       sum(float(row['cost']) for row in selected) / len(selected))
                self.assertEqual(time_fields(group), set(TIMING_FIELDS))
                self.assertAlmostEqual(float(group['path_planning_seconds_total']),
                                       sum(float(row['path_planning_seconds']) for row in selected))
            report = (output / 'REPORT.md').read_text()
            self.assertIn('推理设备', report)
            self.assertIn('按机器人数量分组', report)
            self.assertIn('平均每样本推理时间', report)
            self.assertIn('路径规划时间', report)
            self.assertNotIn('完整流程的墙钟计时', report)
            self.assertNotIn('样本阶段累计计时', report)


if __name__ == '__main__':
    unittest.main()
