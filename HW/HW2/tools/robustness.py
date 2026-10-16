"""穩健性檢查：失敗數、碰撞數、最壞步數使用率、每張耗時，以及幾張刻意刁難的極端地圖。

用法（在 HW2/ 底下）：
    python3 tools/robustness.py DIR [DIR ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from local_score import Agent, run_map  # noqa: E402
from environment import MapDefinition  # noqa: E402


def edge_maps() -> list[MapDefinition]:
    """手工設計的極端地圖：空地、目標在起點附近、蛇形走廊、梳子狀死路。"""
    def mk(name, grid, start, goal):
        d = MapDefinition(name, len(grid), len(grid[0]), tuple(grid), start, goal, "edge")
        d.validate()
        return d

    empty = ["." * 20] * 20
    snake = []
    for r in range(20):
        if r % 2 == 0:
            snake.append("." * 20)
        elif r % 4 == 1:
            snake.append("#" * 19 + ".")
        else:
            snake.append("." + "#" * 19)
    comb = ["." * 20] + ["." + "".join("#."[c % 2] for c in range(19)) for _ in range(19)]
    return [
        mk("empty_corner", empty, (0, 0), (19, 19)),
        mk("empty_goal_near", empty, (10, 10), (10, 13)),
        mk("goal_behind_start", empty, (0, 18), (0, 15)),
        mk("snake", snake, (0, 0), (19, 0)),
        mk("comb_last_tooth", comb, (0, 0), (19, 18)),
        mk("comb_first_tooth", comb, (0, 0), (19, 2)),
    ]


def main():
    agent = Agent()
    total = fails = collisions = 0
    worst = (0.0, "")
    t0 = time.time()
    for d in (Path(p) for p in sys.argv[1:]):
        for path in sorted(d.glob("*.json")):
            definition = MapDefinition.from_json(path)
            r = run_map(agent, definition)
            total += 1
            fails += not r["success"]
            collisions += r["collisions"]
            usage = r["steps"] / (4 * definition.rows * definition.cols)
            if usage > worst[0]:
                worst = (usage, f"{d.name}/{path.stem}")
    if total:
        print(f"maps={total} fails={fails} collisions={collisions} "
              f"time/map={(time.time() - t0) / total * 1000:.1f}ms")
        print(f"worst step-limit usage: {worst[0]:.1%} ({worst[1]})")
    for definition in edge_maps():
        r = run_map(agent, definition)
        limit = 4 * definition.rows * definition.cols
        print(f"{definition.name:<20} ok={r['success']} steps={r['steps']:4}/{limit} "
              f"coll={r['collisions']} opt={r['optimal']:3} score={r['score']:.2f}")


if __name__ == "__main__":
    main()
