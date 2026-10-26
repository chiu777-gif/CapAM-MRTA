"""Flushable CSV metrics, independent of console printing and model state."""
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import time


def add_metrics_arguments(parser):
    parser.add_argument('--metrics_dir', default=None,
                        help='CSV metrics directory; defaults to the run/result metrics directory')
    parser.add_argument('--metrics_step', type=int, default=1,
                        help='Record every N batches; 0 disables batch rows, summaries are always recorded')


class CostAccumulator:
    """Sample-weighted summaries, including an uneven final batch."""
    def __init__(self):
        self.count = 0
        self.total = 0.
        self.total_squared = 0.

    def update(self, costs):
        values = costs.detach().double()
        self.count += values.numel()
        self.total += values.sum().item()
        self.total_squared += values.square().sum().item()

    def summary(self):
        if not self.count:
            raise ValueError('Cannot summarize an empty set of costs')
        mean = self.total / self.count
        variance = max(0., (self.total_squared - self.total ** 2 / self.count) /
                       (self.count - 1)) if self.count > 1 else 0.
        return {'n_samples': self.count, 'mean_cost': mean, 'mean_reward': -mean,
                'cost_std': math.sqrt(variance), 'cost_stderr': math.sqrt(variance / self.count)}


class MetricsLogger:
    FIELDS = ('timestamp_utc', 'elapsed_seconds', 'run', 'epoch', 'batch', 'step',
              'n_samples', 'mean_cost', 'mean_reward', 'cost_std', 'cost_stderr',
              'actor_loss', 'total_loss', 'negative_log_likelihood', 'grad_norm',
              'clipped_grad_norm', 'learning_rate', 'duration_seconds',
              'completion_rate', 'decode_strategy', 'width', 'model_path')
    STREAMS = ('train_batch', 'train_epoch', 'validation_batch', 'validation_epoch',
               'test_batch', 'test_summary')

    def __init__(self, directory, run, batch_interval=1):
        if batch_interval < 0:
            raise ValueError('metrics_step must be nonnegative')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run = run
        self.batch_interval = batch_interval
        self.started = time.perf_counter()
        self._files = {}

    def should_log_batch(self, completed_batches):
        return self.batch_interval > 0 and completed_batches % self.batch_interval == 0

    def write(self, stream, **values):
        if stream not in self.STREAMS:
            raise ValueError('Unknown metrics stream: {}'.format(stream))
        if stream not in self._files:
            path = self.directory / (stream + '.csv')
            has_header = path.exists() and path.stat().st_size > 0
            if has_header:
                with path.open(newline='', encoding='utf-8') as existing:
                    if tuple(next(csv.reader(existing))) != self.FIELDS:
                        raise ValueError('Existing metrics file has a different schema: {}'.format(path))
            handle = path.open('a', newline='', encoding='utf-8')
            writer = csv.DictWriter(handle, fieldnames=self.FIELDS)
            self._files[stream] = handle, writer
            if not has_header:
                writer.writeheader()
        handle, writer = self._files[stream]
        row = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
               'elapsed_seconds': time.perf_counter() - self.started, 'run': self.run}
        row.update(values)
        writer.writerow(row)
        handle.flush()  # Allow tail/plotting while nohup training is still running.

    def close(self):
        for handle, _ in self._files.values():
            handle.close()
        self._files.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
