"""Static obstacle-aware navigation and all-pairs caches in normalized coordinates."""

from collections import deque
from functools import lru_cache
import heapq
import math

import numpy as np


@lru_cache(maxsize=32)
def largest_free_component(rows):
    """Use four-connected cells so robots cannot cross touching obstacle corners."""
    height, width = len(rows), len(rows[0])
    remaining = {(r, c) for r, row in enumerate(rows) for c, value in enumerate(row) if value == '.'}
    largest = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, queue = [start], deque([start])
        while queue:
            r, c = queue.popleft()
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.append(neighbor)
                    queue.append(neighbor)
        if len(component) > len(largest):
            largest = component
    return tuple(sorted(largest))


class GridPathPlanner:
    VERSION = 1

    def __init__(self, rows=(), obstacles=(), subdivisions=10):
        if isinstance(subdivisions, bool) or not isinstance(subdivisions, (int, np.integer)) or subdivisions < 1:
            raise ValueError('path_subdivisions 必须为正整数')
        self.obstacles = np.asarray(obstacles, dtype=float).reshape(-1, 4)
        self.height = (len(rows) if rows else 10) * subdivisions
        self.width = (len(rows[0]) if rows else 10) * subdivisions
        if rows:
            blocked = np.array([[value == '#' for value in row] for row in rows])
            self.blocked = np.repeat(np.repeat(blocked, subdivisions, axis=0), subdivisions, axis=1)
        else:
            # Conservative rasterization for rectangular maps without a character grid.
            self.blocked = np.zeros((self.height, self.width), dtype=bool)
            for xmin, ymin, xmax, ymax in self.obstacles:
                xs = (np.arange(self.width) / self.width < xmax) & ((np.arange(self.width) + 1) / self.width > xmin)
                ys = (1 - np.arange(self.height) / self.height > ymin) & (1 - (np.arange(self.height) + 1) / self.height < ymax)
                self.blocked |= ys[:, None] & xs[None, :]

    def segment_is_free(self, start, goal):
        """Closed rectangles: touching an obstacle edge or corner is blocked."""
        if not len(self.obstacles):
            return True
        start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
        direction = goal - start
        low, high = np.zeros(len(self.obstacles)), np.ones(len(self.obstacles))
        possible = np.ones(len(self.obstacles), dtype=bool)
        for axis in range(2):
            minimum, maximum = self.obstacles[:, axis], self.obstacles[:, axis + 2]
            if direction[axis] == 0:
                possible &= (start[axis] >= minimum) & (start[axis] <= maximum)
            else:
                first = (minimum - start[axis]) / direction[axis]
                second = (maximum - start[axis]) / direction[axis]
                low = np.maximum(low, np.minimum(first, second))
                high = np.minimum(high, np.maximum(first, second))
        return not np.any(possible & (low <= high))

    def _center(self, cell):
        r, c = cell
        return np.array([(c + 0.5) / self.width, 1 - (r + 0.5) / self.height])

    def _cell(self, point):
        x, y = point
        if not np.all(np.isfinite(point)) or not (0 <= x <= 1 and 0 <= y <= 1):
            raise ValueError(f'导航节点坐标超出 [0,1] 地图：{point}')
        cell = (min(int((1 - y) * self.height), self.height - 1),
                min(int(x * self.width), self.width - 1))
        if not self.segment_is_free(point, point):
            raise ValueError(f'导航节点位于障碍物内或边界上：{point}')
        if not self.blocked[cell] and self.segment_is_free(point, self._center(cell)):
            return cell
        # Rounding can select the wrong side of a grid line near an obstacle.
        r, c = cell
        candidates = [(rr, cc) for rr in range(max(0, r - 1), min(self.height, r + 2))
                      for cc in range(max(0, c - 1), min(self.width, c + 2))]
        candidates.sort(key=lambda value: float(np.linalg.norm(self._center(value) - point)))
        for candidate in candidates:
            if not self.blocked[candidate] and self.segment_is_free(point, self._center(candidate)):
                return candidate
        raise ValueError(f'导航节点无法连接到规划网格，请提高 path_subdivisions：{point}')

    def validate_point(self, point):
        self._cell(np.asarray(point, dtype=float))

    def plan(self, start, goal):
        start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
        origin, target = self._cell(start), self._cell(goal)
        if self.segment_is_free(start, goal):
            path = np.array([start, goal])
            return path, float(np.linalg.norm(goal - start))
        scores, previous = {origin: 0.0}, {}
        queue = [(math.hypot((origin[1] - target[1]) / self.width,
                             (origin[0] - target[0]) / self.height), 0.0, origin)]
        while queue:
            _, cost, cell = heapq.heappop(queue)
            if cost > scores[cell]:
                continue
            if cell == target:
                cells = [cell]
                while cells[-1] != origin:
                    cells.append(previous[cells[-1]])
                points = [start] + [self._center(value) for value in reversed(cells)] + [goal]
                # Only remove bends when the entire replacement segment is collision-free.
                smooth, index = [points[0]], 0
                while index < len(points) - 1:
                    following = len(points) - 1
                    while following > index + 1 and not self.segment_is_free(points[index], points[following]):
                        following -= 1
                    if not self.segment_is_free(points[index], points[following]):
                        raise ValueError('规划网格边穿过障碍物，请提高 path_subdivisions')
                    smooth.append(points[following])
                    index = following
                path = np.asarray(smooth)
                return path, float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
            r, c = cell
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.height and 0 <= nc < self.width) or self.blocked[nr, nc]:
                    continue
                if dr and dc and (self.blocked[r + dr, c] or self.blocked[r, c + dc]):
                    continue
                neighbor = (nr, nc)
                step = math.hypot(dc / self.width, dr / self.height)
                new_cost = cost + step
                if new_cost < scores.get(neighbor, math.inf):
                    scores[neighbor], previous[neighbor] = new_cost, cell
                    heuristic = math.hypot((nc - target[1]) / self.width, (nr - target[0]) / self.height)
                    heapq.heappush(queue, (new_cost + heuristic, new_cost, neighbor))
        raise ValueError(f'障碍物阻断路径，节点不可达：{start.tolist()} -> {goal.tolist()}')


class NavigationCache:
    """Plan once for every node pair; position queries never run path search."""

    def __init__(self, nodes, planner):
        self.positions = {key: np.asarray(value, dtype=float).copy() for key, value in nodes.items()}
        self.coordinate_ids = {tuple(point): key for key, point in self.positions.items()}
        self.paths, self.distances, self.relative_positions = {}, {}, {}
        keys = list(nodes)
        for index, first in enumerate(keys):
            for second in keys[index:]:
                try:
                    path, distance = planner.plan(self.positions[first], self.positions[second])
                except ValueError as error:
                    raise ValueError(f'节点 {first} 与 {second} 的路径规划失败：{error}') from error
                for a, b, route in ((first, second, path), (second, first, path[::-1].copy())):
                    self.paths[a, b] = route
                    self.distances[a, b] = distance
                    self.relative_positions[a, b] = self.positions[a] - self.positions[b]

    def pair(self, start, goal):
        try:
            return self.coordinate_ids[tuple(start)], self.coordinate_ids[tuple(goal)]
        except KeyError as error:
            raise ValueError('位置不在节点缓存中；修改坐标后请调用 prepare_navigation(force=True)') from error


def points_along_path(path, distances):
    """Interpolate positions/headings by arc length, including zero-length legs."""
    distances = np.atleast_1d(distances).astype(float)
    segments = np.diff(path, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    valid = lengths > 0
    if not np.any(valid):
        return np.tile([*path[-1], 0.0], (len(distances), 1))
    starts, segments, lengths = path[:-1][valid], segments[valid], lengths[valid]
    cumulative = np.cumsum(lengths)
    distances = np.clip(distances, 0, cumulative[-1])
    index = np.minimum(np.searchsorted(cumulative, distances, side='right'), len(lengths) - 1)
    before = cumulative[index] - lengths[index]
    positions = starts[index] + ((distances - before) / lengths[index])[:, None] * segments[index]
    headings = np.arctan2(segments[index, 1], segments[index, 0])
    return np.column_stack((positions, headings))


def point_along_path(path, distance):
    return points_along_path(path, [distance])[0]
