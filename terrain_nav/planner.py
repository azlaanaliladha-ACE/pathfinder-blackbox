"""Path planning on the image-space cost grid.

Start = bottom-centre cell (in front of the rover). Dijkstra over 8-connected
cells; the goal is the cheapest reachable cell in the farthest traversable row.
"""
import heapq
import math

import numpy as np

import config
from terrain_nav.obstacles import Obstacle

MOVES = [(-1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, 0), (1, -1), (1, 1)]


def build_cost_grid(terrain_cost: np.ndarray, obstacles: list[Obstacle], frame_shape) -> np.ndarray:
    """Terrain cost with obstacle cells (plus inflation) made impassable."""
    cost = terrain_cost.astype(float).copy()
    cost[:config.PLAN_FIRST_ROW] = math.inf          # horizon band is never a drive target
    rows, cols = cost.shape
    ch, cw = frame_shape[0] / rows, frame_shape[1] / cols
    inf_r = config.OBSTACLE_INFLATION

    for ob in obstacles:
        x1, y1, x2, y2 = ob.box
        r2 = min(int(y2 // ch), rows - 1)
        r1 = max(int(y1 // ch), r2 - config.OBSTACLE_FOOTPRINT_ROWS + 1)   # only where it meets the ground
        c1, c2 = int(x1 // cw), min(int(x2 // cw), cols - 1)
        # soft penalty ring, then hard block
        cost[max(r1-inf_r-1, 0):r2+inf_r+2, max(c1-inf_r-1, 0):c2+inf_r+2] += config.OBSTACLE_NEAR_PENALTY
        cost[max(r1-inf_r, 0):r2+inf_r+1, max(c1-inf_r, 0):c2+inf_r+1] = math.inf
    return cost


def plan(cost: np.ndarray) -> list[tuple[int, int]] | None:
    """Return the list of (row, col) cells from start to goal, or None if boxed in."""
    rows, cols = cost.shape
    free = [c for c in range(cols) if not math.isinf(cost[rows - 1, c])]
    if not free:
        return None
    start = (rows - 1, min(free, key=lambda c: abs(c - cols // 2)))   # closest to centre

    dist = np.full(cost.shape, math.inf)
    prev: dict[tuple[int, int], tuple[int, int]] = {}
    dist[start] = 0.0
    heap = [(0.0, start)]
    while heap:
        d, (r, c) = heapq.heappop(heap)
        if d > dist[r, c]:
            continue
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not math.isinf(cost[nr, nc]):
                nd = d + cost[nr, nc] * math.hypot(dr, dc)
                if nd < dist[nr, nc]:
                    dist[nr, nc], prev[(nr, nc)] = nd, (r, c)
                    heapq.heappush(heap, (nd, (nr, nc)))

    reachable = np.argwhere(np.isfinite(dist))
    top_row = reachable[:, 0].min()
    candidates = [tuple(p) for p in reachable if p[0] == top_row]
    goal = min(candidates, key=lambda p: dist[p])

    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    return path[::-1]
