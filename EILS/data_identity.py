"""Fingerprint exactly the task/agent files consumed by the TAPTC loader."""
import hashlib
import json
from pathlib import Path

try:
    from .solver import sha256
except ImportError:
    from solver import sha256


def dataset_identity(path):
    path = Path(path).resolve()
    if path.is_file():
        return {"sha256": sha256(path), "hash_method": "single_file_sha256"}
    if not path.is_dir():
        raise FileNotFoundError(path)
    tasks = sorted(path.glob("group[12]/r*.txt"))
    if not tasks:
        raise ValueError("No original TAPTC group1/group2 task files in {}".format(path))
    sources = set(tasks)
    for task in tasks:
        agent_name = "a" + task.stem.split("a", 1)[1] + ".txt"
        sources.add(task.parent.parent / "agent" / agent_name)
    files = [{"path": source.relative_to(path).as_posix(), "sha256": sha256(source)}
             for source in sorted(sources)]
    # Paths make file additions/deletions/renames detectable, independently of root location.
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"sha256": hashlib.sha256(canonical).hexdigest(),
            "hash_method": "sha256_of_canonical_relative_path_and_file_sha256_manifest",
            "task_files": len(tasks), "agent_files": len(sources) - len(tasks), "files": files}
