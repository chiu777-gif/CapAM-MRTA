"""Regression checks for raw TAPTC directory provenance."""
from pathlib import Path
import tempfile
import unittest

from EILS.data_identity import dataset_identity
from EILS.solver import sha256


class IdentityTests(unittest.TestCase):
    def test_single_file_compatible_with_existing_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.pkl"
            path.write_bytes(b"fixed scenario payload")
            self.assertEqual(dataset_identity(path)["sha256"], sha256(path))

    def test_directory_tracks_consumed_tasks_and_agents_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("group1", "group2", "agent", "n_200"):
                (root / name).mkdir()
            task = root / "group1/r11a2i00.txt"
            task.write_text("task data")
            agent = root / "agent/a2i00.txt"
            agent.write_text("agent data")
            first = dataset_identity(root)
            self.assertEqual(first["task_files"], 1)
            self.assertEqual(first["agent_files"], 1)
            (root / "n_200/unused.pkl").write_bytes(b"other graph size")
            (root / "readme.txt").write_text("metadata")
            self.assertEqual(dataset_identity(root), first)
            agent.write_text("changed agent capacity")
            second = dataset_identity(root)
            self.assertNotEqual(second["sha256"], first["sha256"])
            task.write_text("changed deadline")
            self.assertNotEqual(dataset_identity(root)["sha256"], second["sha256"])

    def test_missing_agent_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "group1").mkdir()
            (root / "group1/r11a2i00.txt").write_text("task data")
            with self.assertRaises(FileNotFoundError):
                dataset_identity(root)


if __name__ == "__main__":
    unittest.main()
