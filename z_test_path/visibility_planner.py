"""Collision-free visibility graphs with Euclidean edges and Dijkstra search."""
import heapq
import math

import numpy as np

from z_test_path.cankao_planner import GridPathPlanner


DEFAULT_VISIBILITY_EPSILON = 1e-6


class VisibilityGraphPlanner:
    """Route through visible, outward-offset obstacle corners in [0,1]^2.

    Obstacles are closed rectangles, as in GridPathPlanner. Offsetting corners
    keeps routes strictly outside their boundaries. Dijkstra finds a shortest
    path on this finite graph, rather than an exact continuous-space geodesic.
    The obstacle graph is built lazily once; endpoint connections and source
    search trees are reused for the many pairs in a scenario.
    """

    VERSION = 1

    def __init__(self, rows, obstacles=(), epsilon=DEFAULT_VISIBILITY_EPSILON):
        if not math.isfinite(epsilon) or epsilon < 4 * np.finfo(float).eps:
            raise ValueError('visibility_epsilon must be finite and >= 4 * float64 machine epsilon')
        if not rows or not rows[0] or epsilon >= .5 / max(len(rows), len(rows[0])):
            raise ValueError('visibility_epsilon must be smaller than half the narrowest ASCII cell edge')
        self.obstacles = np.asarray(obstacles, dtype=float).reshape(-1, 4)
        self.epsilon = float(epsilon)
        self.vertices = None
        self.adjacency = None
        self._connections = {}
        self._source_trees = {}

    # Use exactly the same closed-rectangle collision rule as the grid planner.
    segment_is_free = GridPathPlanner.segment_is_free

    def validate_point(self, point):
        point = np.asarray(point, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all() or ((point < 0) | (point > 1)).any():
            raise ValueError('Navigation point must be a finite 2D coordinate in [0,1]: {}'.format(point))
        if not self.segment_is_free(point, point):
            raise ValueError('Navigation point is inside or on an obstacle boundary: {}'.format(point))

    def _build_graph(self):
        candidates = set()
        for xmin, ymin, xmax, ymax in self.obstacles:
            for x in (xmin - self.epsilon, xmax + self.epsilon):
                for y in (ymin - self.epsilon, ymax + self.epsilon):
                    point = (x, y)
                    if 0 <= x <= 1 and 0 <= y <= 1 and self.segment_is_free(point, point):
                        candidates.add(point)
        self.vertices = np.asarray(sorted(candidates), dtype=float).reshape(-1, 2)
        self.adjacency = [[] for _ in self.vertices]
        for first, start in enumerate(self.vertices):
            for second in range(first + 1, len(self.vertices)):
                goal = self.vertices[second]
                if self.segment_is_free(start, goal):
                    distance = float(np.linalg.norm(goal - start))
                    self.adjacency[first].append((second, distance))
                    self.adjacency[second].append((first, distance))

    def _connect(self, point):
        if self.vertices is None:
            self._build_graph()
        key = tuple(point)
        if key not in self._connections:
            self._connections[key] = [
                (index, float(np.linalg.norm(vertex - point)))
                for index, vertex in enumerate(self.vertices) if self.segment_is_free(point, vertex)
            ]
        return self._connections[key]

    def _shortest_from(self, start):
        key = tuple(start)
        if key not in self._source_trees:
            connections = self._connect(start)
            distances = np.full(len(self.vertices), math.inf)
            previous = np.full(len(self.vertices), -1, dtype=int)
            queue = []
            # Equivalent to adding a source vertex with its visible edges.
            for vertex, distance in connections:
                distances[vertex] = distance
                heapq.heappush(queue, (distance, vertex))
            while queue:
                cost, vertex = heapq.heappop(queue)
                if cost > distances[vertex]:
                    continue
                for neighbor, weight in self.adjacency[vertex]:
                    candidate = cost + weight
                    if candidate < distances[neighbor]:
                        distances[neighbor], previous[neighbor] = candidate, vertex
                        heapq.heappush(queue, (candidate, neighbor))
            self._source_trees[key] = distances, previous
        return self._source_trees[key]

    def plan(self, start, goal):
        start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
        self.validate_point(start)
        self.validate_point(goal)
        if self.segment_is_free(start, goal):
            return np.array([start, goal]), float(np.linalg.norm(goal - start))
        distances, previous = self._shortest_from(start)
        cost, vertex = min(((distances[index] + weight, index) for index, weight in self._connect(goal)),
                           default=(math.inf, -1))
        if not math.isfinite(cost):
            raise ValueError('Visibility graph cannot connect {} -> {}; nodes may be disconnected; '
                             'try a smaller visibility_epsilon'.format(start.tolist(), goal.tolist()))
        chain = []
        while vertex != -1:
            chain.append(self.vertices[vertex])
            vertex = previous[vertex]
        path = np.asarray([start] + chain[::-1] + [goal])
        return path, float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
