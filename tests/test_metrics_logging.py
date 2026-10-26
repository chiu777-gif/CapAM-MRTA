"""CSV correctness, partial batches, resume paths and a real nohup launch."""
import contextlib
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import torch
from options import get_options
from run import run
from utils.metrics_logger import CostAccumulator, MetricsLogger

REPO = Path(__file__).resolve().parents[1]


def rows(directory, stream):
    with (Path(directory) / (stream + '.csv')).open(newline='') as handle:
        return list(csv.DictReader(handle))


def short_train(directory, extra=()):
    opts = get_options(['--no_cuda', '--graph_size', '10', '--n_agents', '2',
                        '--embedding_dim', '32', '--batch_size', '2', '--epoch_size', '5',
                        '--val_size', '3', '--eval_batch_size', '2', '--n_epochs', '1',
                        '--deadline_min', '10', '--deadline_max', '20', '--baseline', 'exponential',
                        '--no_progress_bar', '--log_step', '10000', '--output_dir', str(directory),
                        *extra])
    console = io.StringIO()
    with contextlib.redirect_stdout(console):
        run(opts)
    return opts, console.getvalue()


class MetricsTests(unittest.TestCase):
    def test_sample_weighted_summary_and_reward_definition(self):
        stats = CostAccumulator()
        stats.update(torch.tensor([1., 3.]))
        stats.update(torch.tensor([10.]))
        expected = torch.tensor([1., 3., 10.], dtype=torch.float64)
        summary = stats.summary()
        self.assertEqual(summary['n_samples'], 3)
        self.assertAlmostEqual(summary['mean_cost'], expected.mean().item())
        self.assertAlmostEqual(summary['cost_std'], expected.std().item())
        self.assertEqual(summary['mean_reward'], -summary['mean_cost'])

    def test_flush_and_append_keep_one_header(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('first', 'second'):
                with MetricsLogger(directory, name) as logger:
                    logger.write('train_epoch', epoch=0, step=1, n_samples=2, mean_cost=3., mean_reward=-3.)
                    self.assertEqual(rows(directory, 'train_epoch')[-1]['run'], name)
            self.assertEqual(len(rows(directory, 'train_epoch')), 2)

    def test_every_batch_epoch_and_validation_are_separate_and_weighted(self):
        with tempfile.TemporaryDirectory() as directory:
            opts, console = short_train(directory)
            batches = rows(opts.metrics_dir, 'train_batch')
            self.assertEqual([int(row['step']) for row in batches], [1, 2, 3])
            self.assertEqual([int(row['n_samples']) for row in batches], [2, 2, 1])
            epoch = rows(opts.metrics_dir, 'train_epoch')[0]
            weighted = sum(float(row['mean_cost']) * int(row['n_samples']) for row in batches) / 5
            self.assertAlmostEqual(float(epoch['mean_cost']), weighted)
            self.assertEqual(float(epoch['mean_reward']), -float(epoch['mean_cost']))
            self.assertEqual(console.count('train_batch_id:'), 1)  # log_step does not thin CSV rows.
            validation = rows(opts.metrics_dir, 'validation_batch')
            self.assertEqual([int(row['n_samples']) for row in validation], [2, 1])
            summary = rows(opts.metrics_dir, 'validation_epoch')[0]
            expected = sum(float(row['mean_cost']) * int(row['n_samples']) for row in validation) / 3
            self.assertAlmostEqual(float(summary['mean_cost']), expected)

    def test_batch_interval_zero_keeps_epoch_summaries(self):
        with tempfile.TemporaryDirectory() as directory:
            opts, _ = short_train(directory, ['--metrics_step', '0'])
            self.assertFalse((Path(opts.metrics_dir) / 'train_batch.csv').exists())
            self.assertFalse((Path(opts.metrics_dir) / 'validation_batch.csv').exists())
            self.assertEqual(len(rows(opts.metrics_dir, 'train_epoch')), 1)
            self.assertEqual(len(rows(opts.metrics_dir, 'validation_epoch')), 1)

    def test_resume_uses_new_metrics_directory_and_continues_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            first, _ = short_train(directory, ['--metrics_dir', str(Path(directory) / 'original_metrics')])
            opts = get_options(['--resume', str(Path(first.save_dir) / 'epoch-0.pt'),
                                '--n_epochs', '1', '--output_dir', directory])
            self.assertNotEqual(first.metrics_dir, opts.metrics_dir)
            with contextlib.redirect_stdout(io.StringIO()):
                run(opts)
            self.assertEqual([int(row['step']) for row in rows(opts.metrics_dir, 'train_batch')], [4, 5, 6])
            self.assertEqual(rows(opts.metrics_dir, 'train_epoch')[0]['epoch'], '1')
            self.assertEqual(len(rows(first.metrics_dir, 'train_batch')), 3)

    def test_standalone_evaluation_writes_weighted_batch_and_summary_logs(self):
        from eval import main
        from problems.mrta.config import MRTAConfig
        from problems.mrta.data import RandomMRTADataset, save_training_dataset
        with tempfile.TemporaryDirectory() as directory:
            opts, _ = short_train(directory)
            path = save_training_dataset(RandomMRTADataset(MRTAConfig(graph_size=10), 3, 8),
                                         Path(directory) / 'data.pkl')
            metrics = Path(directory) / 'eval_metrics'
            arguments = ['--model', str(Path(opts.save_dir) / 'epoch-0.pt'), '--datasets', str(path),
                         '--no_cuda', '--no_progress_bar', '--eval_batch_size', '2',
                         '--metrics_dir', str(metrics), '-o', str(Path(directory) / 'result.json')]
            with contextlib.redirect_stdout(io.StringIO()):
                results = main(arguments)
            summary = rows(metrics, 'test_summary')[0]
            self.assertEqual([int(row['n_samples']) for row in rows(metrics, 'test_batch')], [2, 1])
            self.assertAlmostEqual(float(summary['mean_cost']), sum(row['cost'] for row in results) / 3)
            self.assertAlmostEqual(float(summary['completion_rate']), sum(row['completion_rate'] for row in results) / 3)
            # Refusing to overwrite output must not append duplicate metrics first.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main(arguments)
            self.assertEqual(len(rows(metrics, 'test_summary')), 1)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Optional plotting dependency missing')
    def test_plot_saved_metrics(self):
        from plot_metrics import main
        with tempfile.TemporaryDirectory() as directory:
            opts, _ = short_train(directory)
            with contextlib.redirect_stdout(io.StringIO()):
                output = main(['--metrics_dir', opts.metrics_dir])
            self.assertEqual(output.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')

    def test_real_nohup_launch_separates_console_and_metrics_with_space_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = dict(os.environ, PYTHON_BIN=sys.executable, CLOG_DIR=str(root / 'console logs'),
                       MLOG_DIR=str(root / 'metric logs'))
            command = ['bash', str(REPO / 'scripts/train_nohup.sh'), '--no_cuda', '--graph_size', '10',
                       '--n_agents', '2', '--embedding_dim', '32', '--batch_size', '2', '--epoch_size', '2',
                       '--val_size', '2', '--eval_batch_size', '2', '--n_epochs', '1',
                       '--baseline', 'exponential', '--run_name', 'nohup smoke',
                       '--output_dir', str(root / 'outputs')]
            launched = subprocess.run(command, cwd=REPO, env=env, text=True, capture_output=True, timeout=10)
            self.assertEqual(launched.returncode, 0, launched.stderr)
            details = dict(line.split(': ', 1) for line in launched.stdout.splitlines() if ': ' in line)
            pid = int(details['PID'])
            completed = False
            try:
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    console = Path(details['Console']).read_text()
                    if 'Traceback' in console:
                        self.fail(console)
                    if 'Saved ' in console:
                        completed = True
                        break
                    time.sleep(.05)
                self.assertTrue(completed, 'Background job did not finish: ' + console)
                self.assertEqual(Path(details['PID file']).read_text().strip(), str(pid))
                self.assertEqual(len(rows(details['Metrics'], 'train_batch')), 1)
                self.assertEqual(len(rows(details['Metrics'], 'validation_epoch')), 1)
            finally:
                if not completed:
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass


if __name__ == '__main__':
    unittest.main()
