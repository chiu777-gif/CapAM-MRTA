"""Shared validation and selection of Python/NumPy or C++ ILS/EILS search."""
import ctypes
import hashlib
from pathlib import Path
import subprocess
import numpy as np

HERE = Path(__file__).resolve().parent
LIBRARY = HERE / "build" / "libeils.so"
PROFILES = {
    "tight": {"ils_trials": 600, "ils_coefficient": 0.5,
              "eils_iterations": 3000, "eils_coefficient": 1 / 3},
    "slack": {"ils_trials": 600, "ils_coefficient": 0.4,
              "eils_iterations": 5000, "eils_coefficient": 0.125},
}
_library = None


def build():
    source = HERE / "solver.cpp"
    if not LIBRARY.exists() or source.stat().st_mtime_ns > LIBRARY.stat().st_mtime_ns:
        LIBRARY.parent.mkdir(exist_ok=True)
        subprocess.run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC",
                        str(source), "-o", str(LIBRARY)], check=True)
    return LIBRARY


def library():
    global _library
    if _library is None:
        _library = ctypes.CDLL(str(build()))
        array_f = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        array_i = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        _library.solve_taptc.argtypes = [
            ctypes.c_int, ctypes.c_int, array_f, array_f, array_f, array_f, array_f,
            ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_double,
            ctypes.c_uint64, ctypes.c_int, array_f, array_i, array_i, array_i, array_i]
        _library.solve_taptc.restype = ctypes.c_int
    return _library


def prepare(instance):
    """Keep saved normalized units; ratios already give travel/service seconds."""
    m = int(instance["n_agents"])
    arrays = {}
    for name in ("loc", "robots_start_location", "deadline", "workload", "robots_work_capacity"):
        value = instance[name]
        if hasattr(value, "numpy"):
            value = value.numpy()
        arrays[name] = np.ascontiguousarray(value, dtype=np.float64)
    arrays["robots_start_location"] = arrays["robots_start_location"][:m].copy()
    arrays["robots_work_capacity"] = arrays["robots_work_capacity"][:m].copy()
    n = len(arrays["deadline"])
    shapes = {"loc": (n, 2), "robots_start_location": (m, 2), "deadline": (n,),
              "workload": (n,), "robots_work_capacity": (m,)}
    for name, shape in shapes.items():
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise ValueError("Invalid {}: expected finite {}".format(name, shape))
    speed = float(instance["max_speed"])
    if (n < 1 or m < 1 or not np.isfinite(speed) or speed <= 0
            or (arrays["deadline"] <= 0).any() or (arrays["workload"] < 0).any()
            or (arrays["robots_work_capacity"] <= 0).any()):
        raise ValueError("Invalid task count, robot count, speed, deadlines or work values")
    arrays.update(n=n, m=m, speed=speed)
    return arrays


def solve(data, algorithm="eils", profile="tight", seed=20261025,
          service_time="continuous", limit=None, coefficient=None, backend="python"):
    parameters = PROFILES[profile]
    if algorithm not in ("construct", "ils", "eils"):
        raise ValueError("Unknown algorithm")
    if service_time not in ("continuous", "ceil"):
        raise ValueError("Unknown service time mode")
    default_limit = parameters["ils_trials" if algorithm == "ils" else "eils_iterations"]
    default_coefficient = parameters["ils_coefficient" if algorithm == "ils" else "eils_coefficient"]
    limit = default_limit if limit is None else limit
    coefficient = default_coefficient if coefficient is None else coefficient
    if not isinstance(limit, int) or limit < 1 or not 0 < coefficient <= 1:
        raise ValueError("limit must be positive integer; coefficient must be in (0, 1]")
    if not 0 <= seed < 2**64:
        raise ValueError("seed must be in [0, 2**64)")
    if backend == "python":
        try:
            from .python_solver import solve_python
        except ImportError:
            from python_solver import solve_python
        return solve_python(data, algorithm, limit, coefficient, seed, service_time)
    if backend != "cpp":
        raise ValueError("backend must be python or cpp")
    n, m = data["n"], data["m"]
    finish = np.zeros(n, dtype=np.float64)
    robots = np.zeros(n, dtype=np.int32)
    order = np.zeros(n, dtype=np.int32)
    lengths = np.zeros(m, dtype=np.int32)
    stats = np.zeros(3, dtype=np.int32)
    status = library().solve_taptc(
        n, m, data["loc"], data["robots_start_location"], data["deadline"],
        data["workload"], data["robots_work_capacity"], data["speed"],
        {"construct": 0, "ils": 1, "eils": 2}[algorithm], limit, coefficient, seed,
        int(service_time == "ceil"), finish, robots, order, lengths, stats)
    if status:
        raise RuntimeError("Native solver failed with status {}".format(status))
    ends = np.cumsum(lengths)
    routes = [order[start:end].tolist() for start, end in zip(np.r_[0, ends[:-1]], ends)]
    return {"finish_times": finish, "robots": robots, "routes": routes,
            "native_completed_tasks": int(stats[0]), "iterations": int(stats[1]),
            "best_iteration": int(stats[2])}


def validate_solution(data, result, service_time="continuous"):
    """Independent full route replay, outside inference timing."""
    routes = result["routes"]
    flattened = [task for route in routes for task in route]
    if len(routes) != data["m"] or sorted(flattened) != list(range(data["n"])):
        raise ValueError("Routes must allocate every task exactly once to active robots")
    replay = np.zeros(data["n"])
    native_prefix_count = 0
    for robot, route in enumerate(routes):
        previous = data["robots_start_location"][robot]
        time = 0.0
        seen_late = False
        for task in route:
            if result["robots"][task] != robot:
                raise ValueError("Task/robot mapping disagrees with route")
            service = data["workload"][task] / data["robots_work_capacity"][robot]
            if service_time == "ceil":
                service = np.ceil(service)
            time += float(np.linalg.norm(data["loc"][task] - previous)) / data["speed"] + service
            replay[task] = time
            on_time = time <= data["deadline"][task]
            if on_time and seen_late:
                raise ValueError("Unexpected feasible task after residual completion")
            native_prefix_count += int(on_time)
            seen_late |= not on_time
            previous = data["loc"][task]
    if not np.allclose(replay, result["finish_times"], atol=1e-9, rtol=1e-12):
        raise ValueError("Native finish times disagree with independent route replay")
    if native_prefix_count != result["native_completed_tasks"]:
        raise ValueError("Residual completion changed the native search score")
    late = replay > data["deadline"]
    return {"tasks_done": int((~late).sum()),
            "cost": float((replay[late] / data["deadline"][late]).sum())}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
