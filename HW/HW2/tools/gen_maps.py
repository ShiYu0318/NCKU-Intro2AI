"""隨機地圖產生器：模仿公開地圖的三種 style，用來避免只對公開地圖調參。

用法（在 HW2/ 底下）：
    python3 tools/gen_maps.py OUT_DIR [--count N] [--seed S] [--goal far|very_far|uniform]

每種尺寸（10/15/20）× 每種 style（open/dead_ends/loops_dense）各產生 N 張。
同一個 seed 會產生完全相同的地圖，方便重現實驗。
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path

SIZES = (10, 15, 20)
STYLES = ("open", "dead_ends", "loops_dense")
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def open_grid(rng: random.Random, n: int) -> list[list[str]]:
    """零散的牆，沒有外框。"""
    density = rng.uniform(0.10, 0.25)
    return [["#" if rng.random() < density else "." for _ in range(n)] for _ in range(n)]


def perfect_maze(rng: random.Random, n: int) -> list[list[str]]:
    """Recursive backtracker：在奇數座標上挖出一個每對格子只有一條路的迷宮。"""
    grid = [["#"] * n for _ in range(n)]
    cells = [(r, c) for r in range(1, n - 1, 2) for c in range(1, n - 1, 2)]
    start = rng.choice(cells)
    grid[start[0]][start[1]] = "."
    stack = [start]
    while stack:
        r, c = stack[-1]
        options = []
        for dr, dc in DIRS:
            nr, nc = r + 2 * dr, c + 2 * dc
            if 1 <= nr < n - 1 and 1 <= nc < n - 1 and grid[nr][nc] == "#":
                options.append((nr, nc, r + dr, c + dc))
        if not options:
            stack.pop()
            continue
        nr, nc, wr, wc = rng.choice(options)
        grid[wr][wc] = "."
        grid[nr][nc] = "."
        stack.append((nr, nc))
    return grid


def loopy_maze(rng: random.Random, n: int) -> list[list[str]]:
    """完美迷宮 + 隨機打掉一部分內牆，形成大量環路。"""
    grid = perfect_maze(rng, n)
    ratio = rng.uniform(0.20, 0.40)
    for r in range(1, n - 1):
        for c in range(1, n - 1):
            if grid[r][c] != "#":
                continue
            # 只拆「夾在兩條通道中間」的牆，才會真的連出一條環路
            vertical = grid[r - 1][c] == "." and grid[r + 1][c] == "."
            horizontal = grid[r][c - 1] == "." and grid[r][c + 1] == "."
            if (vertical or horizontal) and rng.random() < ratio:
                grid[r][c] = "."
    return grid


def bfs_dist(grid: list[list[str]], start: tuple[int, int]) -> dict:
    n = len(grid)
    dist = {start: 0}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] == "." and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                queue.append((nr, nc))
    return dist


def make_map(rng: random.Random, n: int, style: str, goal_mode: str, name: str) -> dict:
    builder = {"open": open_grid, "dead_ends": perfect_maze, "loops_dense": loopy_maze}[style]
    while True:
        grid = builder(rng, n)
        free = [(r, c) for r in range(n) for c in range(n) if grid[r][c] == "."]
        if len(free) < 10:
            continue
        start = rng.choice(free)
        dist = bfs_dist(grid, start)
        # 規則：目標必須可達，而且不在起點的上下左右（距離 >= 2）
        candidates = sorted((d, cell) for cell, d in dist.items() if d >= 2)
        if len(candidates) < 4:
            continue
        if goal_mode == "far":
            candidates = candidates[len(candidates) // 2 :]
        elif goal_mode == "very_far":
            # 公開地圖的目標落在距離第 83~99 百分位，這個模式最接近
            candidates = candidates[len(candidates) * 4 // 5 :]
        goal = rng.choice(candidates)[1]
        return {
            "name": name,
            "rows": n,
            "cols": n,
            "style": style,
            "grid": ["".join(row) for row in grid],
            "start": list(start),
            "goal": list(goal),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--goal", choices=("far", "very_far", "uniform"), default="far")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for n in SIZES:
        for style in STYLES:
            for i in range(args.count):
                name = f"{style}_{n}_{i:03d}"
                data = make_map(rng, n, style, args.goal, name)
                (out / f"{name}.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"wrote {len(SIZES) * len(STYLES) * args.count} maps to {out}")


if __name__ == "__main__":
    main()
