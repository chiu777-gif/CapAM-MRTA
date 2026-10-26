"""End-to-end checks for test data, completion aggregation and timing output."""
import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch
from nets.attention_model import AttentionModel
from problems.mrta.config import DATA_VERSION
from problems.mrta.data import load_file_dataset
from problems.mrta.problem_mrta import MRTA
from z_test.evaluate import main as evaluate
from z_test.generate_data import main as generate


def quiet(function, args):
    with contextlib.redirect_stdout(io.StringIO()):
        return function(args)


class TestWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        cls.checkpoint_dir = tempfile.TemporaryDirectory()
        root = Path(cls.checkpoint_dir.name)
        model = AttentionModel(32, 32, MRTA, normalization='batch')
        cls.checkpoint = root / 'epoch-0.pt'
        torch.save({'model': model.state_dict()}, cls.checkpoint)
        (root / 'args.json').write_text(json.dumps({
            'problem': 'mrta', 'embedding_dim': 32, 'normalization': 'batch',
            'tanh_clipping': 10., 'network_version': DATA_VERSION, 'schema_version': DATA_VERSION,
            'coordinate_scale': 100., 'workload_scale': 100.,
        }))

    @classmethod
    def tearDownClass(cls):
        cls.checkpoint_dir.cleanup()

    def test_physical_parameters_roundtrip_and_overwrite_protection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'custom.pkl'
            args = ['--graph_size', '3', '--dataset_size', '2', '--n_agents', '2',
                    '--map_size', '200', '--agent_max_speed', '2',
                    '--deadline_min', '150', '--deadline_max', '150',
                    '--workload_min', '20', '--workload_max', '20',
                    '--work_capacity_min', '2', '--work_capacity_max', '2',
                    '--filename', str(path)]
            quiet(generate, args)
            data = load_file_dataset(path)
            self.assertEqual(len(data), 2)
            for item in data:
                self.assertEqual(item['loc'].shape, (3, 2))
                self.assertEqual(item['n_agents'].item(), 2)
                self.assertTrue(((item['loc'] >= 0) & (item['loc'] <= 2)).all())
                torch.testing.assert_close(item['max_speed'], torch.tensor(.02))
                torch.testing.assert_close(item['workload'], torch.full((3,), .2))
                torch.testing.assert_close(item['robots_work_capacity'], torch.full((2,), .02))
                torch.testing.assert_close(item['deadline'], torch.full((3,), 150.))
            metadata = json.loads(path.with_suffix('.json').read_text())
            self.assertEqual(metadata['physical_parameters']['agent_max_speed'], 2.)
            original = path.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                quiet(generate, args)
            self.assertEqual(original, path.read_bytes())
            other = Path(directory) / 'repeat.pkl'
            quiet(generate, args[:-1] + [str(other)])
            for key in data[0]:
                torch.testing.assert_close(data[0][key], load_file_dataset(other)[0][key])

    def test_mixed_sizes_use_task_weighted_completion_and_exclude_warmup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = root / 'data'
            for tasks, deadline, name in [(1, '1000000', 'on_time'), (3, '.001', 'late')]:
                quiet(generate, ['--graph_sizes', str(tasks), '--dataset_size', '1', '--n_agents', '1',
                                 '--workload_min', '20', '--workload_max', '20',
                                 '--work_capacity_min', '1', '--work_capacity_max', '1',
                                 '--deadline_min', deadline, '--deadline_max', deadline,
                                 '--filename', str(inputs / (name + '.pkl'))])
            output = root / 'results'
            summary = quiet(evaluate, ['--model', str(self.checkpoint), '--data', str(inputs),
                                       '--output_dir', str(output), '--no_cuda', '--cpu_threads', '1',
                                       '--warmup_batches', '2', '--no_progress_bar'])
            self.assertEqual(summary['device'], 'cpu')
            self.assertEqual(summary['completed_tasks'], 1)
            self.assertEqual(summary['total_tasks'], 4)
            self.assertEqual(summary['completion_percent'], 25.)
            self.assertEqual(summary['mean_scenario_completion_rate'], .5)
            self.assertEqual(summary['inference_batches'], 2)
            with (output / 'batches.csv').open(newline='') as handle:
                batches = list(csv.DictReader(handle))
            self.assertAlmostEqual(summary['inference_seconds_total'],
                                   sum(float(row['inference_seconds']) for row in batches))
            self.assertGreater(summary['inference_seconds_per_scenario'], 0)
            self.assertEqual(json.loads((output / 'summary.json').read_text())['total_tasks'], 4)
            self.assertTrue((output / 'groups.csv').is_file())

    def test_uneven_batch_does_not_duplicate_inference_time(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'data.pkl'
            quiet(generate, ['--graph_sizes', '2', '--dataset_size', '3', '--n_agents', '1',
                             '--deadline_min', '1000000', '--deadline_max', '1000000', '--filename', str(path)])
            summary = quiet(evaluate, ['--model', str(self.checkpoint), '--data', str(path),
                                       '--output_dir', str(root / 'results'), '--device', 'cpu',
                                       '--cpu_threads', '1', '--eval_batch_size', '2', '--no_progress_bar'])
            self.assertEqual(summary['scenarios'], 3)
            self.assertEqual(summary['inference_batches'], 2)
            self.assertEqual(summary['completion_rate'], 1.)
            with (root / 'results/batches.csv').open(newline='') as handle:
                batches = list(csv.DictReader(handle))
            self.assertEqual([int(row['scenarios']) for row in batches], [2, 1])
            self.assertAlmostEqual(summary['inference_seconds_total'],
                                   sum(float(row['inference_seconds']) for row in batches))

    def test_scale_mismatch_is_rejected_before_results_are_written(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'data.pkl'
            quiet(generate, ['--graph_sizes', '2', '--dataset_size', '1', '--coordinate_scale', '50',
                             '--filename', str(path)])
            error = io.StringIO()
            with contextlib.redirect_stderr(error), self.assertRaises(SystemExit):
                quiet(evaluate, ['--model', str(self.checkpoint), '--data', str(path),
                                 '--output_dir', str(root / 'results'), '--device', 'cpu', '--cpu_threads', '1'])
            self.assertIn('coordinate_scale', error.getvalue())
            self.assertFalse((root / 'results/summary.json').exists())

    def test_sampling_is_repeatable_across_warmup_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'data.pkl'
            quiet(generate, ['--graph_sizes', '8', '--dataset_size', '2', '--n_agents', '2',
                             '--deadline_min', '50', '--deadline_max', '200', '--filename', str(path)])
            outputs = []
            for warmup in (0, 2):
                output = root / ('warmup_' + str(warmup))
                quiet(evaluate, ['--model', str(self.checkpoint), '--data', str(path),
                                 '--output_dir', str(output), '--device', 'cpu', '--cpu_threads', '1',
                                 '--decode_strategy', 'sample', '--width', '3', '--eval_batch_size', '2',
                                 '--max_calc_batch_size', '4', '--warmup_batches', str(warmup),
                                 '--no_progress_bar'])
                with (output / 'scenarios.csv').open(newline='') as handle:
                    outputs.append([(row['cost'], row['tasks_done']) for row in csv.DictReader(handle)])
            self.assertEqual(outputs[0], outputs[1])

    def test_both_cli_scripts_work_outside_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated = subprocess.run([
                sys.executable, str(REPO / 'z_test/generate_data.py'), '--graph_sizes', '2', '4',
                '--n_agents', '1', '--dataset_size', '1', '--output_dir', str(root / 'data'),
            ], cwd=root, capture_output=True, text=True, timeout=30)
            self.assertEqual(generated.returncode, 0, generated.stderr)
            self.assertEqual(len(list((root / 'data').glob('*.pkl'))), 2)
            evaluated = subprocess.run([
                sys.executable, str(REPO / 'z_test/evaluate.py'), '--model', str(self.checkpoint),
                '--data', str(root / 'data'), '--output_dir', str(root / 'results'),
                '--device', 'cpu', '--cpu_threads', '1', '--no_progress_bar',
            ], cwd=root, capture_output=True, text=True, timeout=30)
            self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
            self.assertIn('Task completion rate:', evaluated.stdout)
            self.assertIn('Total inference time:', evaluated.stdout)

    def test_am_checkpoint_greedy_and_sampling_evaluation(self):
        self.check_encoder_evaluation('am')

    def test_mamba_checkpoint_greedy_and_sampling_evaluation(self):
        self.check_encoder_evaluation('mamba')

    def check_encoder_evaluation(self, encoder):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / encoder
            model_dir.mkdir()
            model = AttentionModel(32, 32, MRTA, encoder=encoder, n_encode_layers=2)
            torch.save({'model': model.state_dict()}, model_dir / 'epoch-0.pt')
            (model_dir / 'args.json').write_text(json.dumps({
                'problem': 'mrta', 'encoder': encoder, 'n_encode_layers': 2,
                'embedding_dim': 32, 'normalization': 'batch', 'tanh_clipping': 10.,
                'network_version': DATA_VERSION, 'schema_version': DATA_VERSION,
            }))
            path = root / 'data.pkl'
            quiet(generate, ['--graph_sizes', '5', '--dataset_size', '3', '--n_agents', '2',
                             '--filename', str(path)])
            for strategy, width in [('greedy', 1), ('sample', 3)]:
                with self.subTest(strategy=strategy):
                    output = root / strategy
                    summary = quiet(evaluate, ['--model', str(model_dir), '--data', str(path),
                                               '--output_dir', str(output), '--device', 'cpu',
                                               '--cpu_threads', '1', '--eval_batch_size', '2',
                                               '--decode_strategy', strategy, '--width', str(width),
                                               '--max_calc_batch_size', '4', '--no_progress_bar'])
                    self.assertEqual((summary['encoder'], summary['n_encode_layers']), (encoder, 2))
                    self.assertEqual(summary['scenarios'], 3)
                    self.assertEqual(summary['total_tasks'], 15)
                    self.assertEqual(json.loads((output / 'summary.json').read_text())['encoder'], encoder)


if __name__ == '__main__':
    unittest.main()
