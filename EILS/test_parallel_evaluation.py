"""Serial/process parity, persisted result audit, and the spawned CLI path."""
import contextlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from EILS.compare_backends import read_routes, read_rows
from EILS.evaluate import main as evaluate
from EILS.verify_results import verify


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data/mrta_gini"


class ParallelEvaluationTests(unittest.TestCase):
    def compare_runs(self, directory, args, expected_solutions):
        outputs = [Path(directory) / "serial", Path(directory) / "parallel"]
        summaries = []
        original_environment = dict(os.environ)
        with contextlib.redirect_stdout(io.StringIO()):
            for workers, output in zip((1, 2), outputs):
                summary = evaluate(["--data", str(DATA), "--progress_every", "0",
                                    "--workers", str(workers), "--output_dir", str(output)] + args)
                self.assertEqual(summary["timing"]["workers"], workers)
                self.assertEqual(verify(output)["solutions_replayed"], expected_solutions)
                summaries.append(summary)
        self.assertEqual(dict(os.environ), original_environment)
        serial_rows, parallel_rows = [read_rows(output) for output in outputs]
        self.assertEqual(set(serial_rows), set(parallel_rows))
        for key, reference in serial_rows.items():
            actual = parallel_rows[key]
            # Wall/CPU timings vary; all identifiers, seeds, metrics and search stats must agree.
            for field in reference:
                if field not in ("inference_seconds", "cpu_seconds"):
                    self.assertEqual(reference[field], actual[field], (key, field))
        self.assertEqual(read_routes(outputs[0]), read_routes(outputs[1]))
        for name, reference in summaries[0]["models"].items():
            actual = summaries[1]["models"][name]
            for field in ("completed_tasks", "total_tasks", "completion_percent", "mean_cost", "mean_iterations"):
                self.assertEqual(reference[field], actual[field], (name, field))
        report = (outputs[1] / "REPORT.md").read_text()
        self.assertIn("CPU 工作进程数：2", report)
        self.assertIn("不是整批耗时除以求解次数", report)
        return summaries

    def test_fixed_parallel_matches_serial_with_repeats_offset_and_seed_wrap(self):
        with tempfile.TemporaryDirectory(dir=str(HERE)) as directory:
            self.compare_runs(directory, ["--backend", "python", "--num_samples", "3", "--offset", "47",
                                          "--repeats", "2", "--seed", str(2**64 - 2), "--limit", "6"], 24)

    def test_group_matched_parallel_matches_serial_for_both_backends(self):
        for backend in ("python", "cpp"):
            if backend == "cpp" and not shutil.which("g++"):
                continue
            with self.subTest(backend=backend), tempfile.TemporaryDirectory(dir=str(HERE)) as directory:
                summaries = self.compare_runs(directory, ["--backend", backend, "--num_samples", "2",
                                                          "--offset", "47", "--repeats", "2", "--limit", "3",
                                                          "--match_taptc_groups", "--service_time", "ceil"], 8)
                for summary in summaries:
                    self.assertEqual(summary["profile_mode"], "taptc_matched")
                    for model in summary["matched_models"].values():
                        self.assertEqual(model["scenarios"], 4)
                        self.assertEqual(set(model["by_deadline_group"]), {"tight", "slack"})

    def test_spawned_cli_progress_and_worker_cap(self):
        with tempfile.TemporaryDirectory(dir=str(HERE)) as directory:
            output = Path(directory) / "run"
            result = subprocess.run([sys.executable, str(HERE / "evaluate.py"), "--data", str(DATA),
                                     "--num_samples", "2", "--algorithms", "eils", "--profiles", "tight",
                                     "--limit", "2", "--workers", "4", "--progress_every", "1",
                                     "--output_dir", str(output)],
                                    capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Loading dataset:", result.stdout)
            self.assertIn("backend=python, workers=2", result.stdout)
            self.assertIn("1/2 scenarios, 1/2 runs", result.stdout)
            self.assertIn("2/2 scenarios, 2/2 runs", result.stdout)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(verify(output)["solutions_replayed"], 2)

    def test_reject_nonpositive_workers(self):
        for workers in ("0", "-1"):
            with self.subTest(workers=workers), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    evaluate(["--workers", workers])
                self.assertEqual(caught.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
