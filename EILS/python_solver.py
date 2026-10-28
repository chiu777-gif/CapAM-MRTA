"""ILS/EILS search in Python and NumPy, without a compiled solver extension.

The candidate table, task/robot/position tie order, MT19937-64 draws and
stopping rules follow solver.cpp. NumPy evaluates the insertion candidates
for one robot together; search control and route operations remain Python.
"""
import math

import numpy as np


class MT19937_64:
    """The standard 64-bit Mersenne Twister used by std::mt19937_64."""
    _MASK = (1 << 64) - 1

    def __init__(self, seed):
        state = [int(seed) & self._MASK]
        for index in range(1, 312):
            previous = state[-1]
            state.append((6364136223846793005 * (previous ^ (previous >> 62)) + index) & self._MASK)
        self.state = state
        self.index = 312

    def __call__(self):
        if self.index == 312:
            state = self.state
            for index in range(312):
                value = ((state[index] & 0xFFFFFFFF80000000)
                         | (state[(index + 1) % 312] & 0x7FFFFFFF))
                shifted = value >> 1
                if value & 1:
                    shifted ^= 0xB5026F5AA96619E9
                state[index] = state[(index + 156) % 312] ^ shifted
            self.index = 0
        value = self.state[self.index]
        self.index += 1
        value ^= (value >> 29) & 0x5555555555555555
        value ^= (value << 17) & 0x71D67FFFEDA60000
        value ^= (value << 37) & 0xFFF7EEE000000000
        value ^= value >> 43
        return value


class Search:
    def __init__(self, data, service_time):
        self.n, self.m = data["n"], data["m"]
        self.deadline = data["deadline"]
        xy = data["loc"]
        starts = data["robots_start_location"]
        speed = data["speed"]
        # NumPy hypot matches the C++ libm hypot in this environment. Python
        # math.hypot uses a different norm algorithm and can shift ties by 1 ULP.
        if "task_to_task" in data:
            self.travel = data["task_to_task"] / speed
            self.start_travel = data["robot_to_task"] / speed
        else:
            self.travel = np.hypot(xy[:, None, 0] - xy[None, :, 0],
                                   xy[:, None, 1] - xy[None, :, 1]) / speed
            self.start_travel = np.hypot(starts[:, None, 0] - xy[None, :, 0],
                                         starts[:, None, 1] - xy[None, :, 1]) / speed
        self.service = data["workload"][None, :] / data["robots_work_capacity"][:, None]
        if service_time == "ceil":
            self.service = np.ceil(self.service)
        self.tours = [[] for _ in range(self.m)]
        self.finish = [[] for _ in range(self.m)]
        self.slack = [[] for _ in range(self.m)]
        self.used = np.zeros(self.n, dtype=bool)
        # C order flattens as task, then robot, matching the C++ selection loop.
        self.cache_shift = np.full((self.n, self.m), np.inf)
        self.cache_position = np.full((self.n, self.m), -1, dtype=np.int32)

    def leg(self, robot, previous, task):
        return float(self.start_travel[robot, task] if previous < 0 else self.travel[previous, task])

    def times(self, robot):
        tour = self.tours[robot]
        finished = []
        time = 0.0
        previous = -1
        for task in tour:
            time += self.leg(robot, previous, task) + float(self.service[robot, task])
            finished.append(time)
            previous = task
        slack = [0.0] * len(tour)
        minimum = math.inf
        for position in range(len(tour) - 1, -1, -1):
            minimum = min(minimum, float(self.deadline[tour[position]]) - finished[position])
            slack[position] = minimum
        self.finish[robot], self.slack[robot] = finished, slack

    def candidates(self, robot):
        tour = self.tours[robot]
        size = len(tour)
        incoming = np.empty((self.n, size + 1))
        incoming[:, 0] = self.start_travel[robot]
        if size:
            incoming[:, 1:] = self.travel[tour, :].T
        increment = incoming + self.service[robot, :, None]
        completion = np.array([0.0] + self.finish[robot])[None, :] + increment
        feasible = (completion <= self.deadline[:, None]) & ~self.used[:, None]
        shift = increment.copy()
        if size:
            old_legs = np.array([self.leg(robot, tour[p - 1] if p else -1, task)
                                 for p, task in enumerate(tour)])
            # Preserve subtraction-then-addition order from Eq. (6).
            shift[:, :size] += self.travel[:, tour] - old_legs[None, :]
            feasible[:, :size] &= shift[:, :size] <= np.array(self.slack[robot])[None, :]
        shift[~feasible] = np.inf
        positions = np.argmin(shift, axis=1)
        minimum = shift[np.arange(self.n), positions]
        self.cache_shift[:, robot] = minimum
        self.cache_position[:, robot] = np.where(np.isfinite(minimum), positions, -1)

    def construct(self):
        score = sum(map(len, self.tours))
        for robot in range(self.m):
            self.times(robot)
            self.candidates(robot)
        while score < self.n:
            flat = int(np.argmin(self.cache_shift))
            task, robot = divmod(flat, self.m)
            if not math.isfinite(float(self.cache_shift[task, robot])):
                break
            position = int(self.cache_position[task, robot])
            self.tours[robot].insert(position, task)
            self.used[task] = True
            self.cache_shift[task, :] = np.inf
            score += 1
            self.times(robot)
            self.candidates(robot)
        return score

    def shake(self, number, start, random_start, rng):
        for robot, tour in enumerate(self.tours):
            size = len(tour)
            if not size:
                continue
            first = rng() % size if random_start else start % size
            count = min(number, size)
            kept = []
            for position, task in enumerate(tour):
                if (position - first + size) % size < count:
                    self.used[task] = False
                else:
                    kept.append(task)
            self.tours[robot] = kept

    def completed_cost(self):
        """Score the full solution without changing the feasible search state."""
        tours = [tour.copy() for tour in self.tours]
        used = self.used.copy()
        try:
            finished = self.complete_all()["finish_times"]
            # Sum in task order, matching the C++ backend's objective comparisons.
            return sum(float(finished[task] / self.deadline[task])
                       for task in range(self.n) if finished[task] > self.deadline[task])
        finally:
            self.tours = tours
            self.used[:] = used

    def complete_all(self):
        self.used[:] = False
        ready = [0.0] * self.m
        previous = [-1] * self.m
        finished = np.zeros(self.n)
        robots = np.zeros(self.n, dtype=np.int32)
        for robot, tour in enumerate(self.tours):
            for task in tour:
                ready[robot] += self.leg(robot, previous[robot], task) + float(self.service[robot, task])
                finished[task], robots[task] = ready[robot], robot
                previous[robot] = task
                self.used[task] = True
        assigned = int(self.used.sum())
        while assigned < self.n:
            robot = min(range(self.m), key=ready.__getitem__)
            increments = ((self.start_travel[robot] if previous[robot] < 0
                           else self.travel[previous[robot]]) + self.service[robot]).copy()
            increments[self.used] = np.inf
            task = int(np.argmin(increments))
            ready[robot] += float(increments[task])
            finished[task], robots[task] = ready[robot], robot
            previous[robot] = task
            self.tours[robot].append(task)
            self.used[task] = True
            assigned += 1
        return {"finish_times": finished, "robots": robots, "routes": self.tours}


def solve_python(data, algorithm, limit, coefficient, seed, service_time, objective="completion"):
    search = Search(data, service_time)
    rng = MT19937_64(seed)
    best = [[] for _ in range(data["m"])]
    best_score, iterations, best_iteration, trials = -1, 0, 0, 0
    best_cost = math.inf
    remove, start = 1, 0
    threshold = max(1, math.floor(data["n"] * coefficient / data["m"]))
    while (iterations < limit if algorithm == "eils" else
           iterations < 1 if algorithm == "construct" else trials < limit):
        iterations += 1
        score = search.construct()
        cost = search.completed_cost() if objective == "cost" else math.inf
        improved = cost < best_cost if objective == "cost" else score > best_score
        if improved:
            best_score = score
            best_cost = cost
            best = [tour.copy() for tour in search.tours]
            best_iteration = iterations
            trials, remove = 0, 1
        else:
            trials += 1
        if (cost == 0 if objective == "cost" else score == data["n"]) or algorithm == "construct":
            break
        if algorithm == "eils":
            count = max(1, math.floor(score * coefficient / data["m"]))
            search.shake(count, 0, True, rng)
        else:
            search.shake(remove, start, False, rng)
            start += remove
            remove += 1
            shortest = min(map(len, search.tours))
            start = start % shortest if shortest > 0 else 0
            if remove >= threshold:
                remove = 1
    search.tours = best
    result = search.complete_all()
    result.update(native_completed_tasks=best_score, iterations=iterations, best_iteration=best_iteration)
    return result
