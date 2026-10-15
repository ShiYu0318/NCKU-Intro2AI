"""比較不同 frontier 評分策略：每個變體是一個 Agent 子類別，只改 _frontier_cost。

用法（在 HW2/ 底下）：
    python3 tools/gen_maps.py /tmp/far --count 30 --seed 1
    python3 tools/gen_maps.py /tmp/uni --count 30 --seed 2 --goal uniform
    python3 tools/experiment.py /tmp/far /tmp/uni
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from local_score import HW2_DIR, Agent, evaluate_dir  # noqa: E402


def unknown_within(agent: Agent, cell, radius: int) -> int:
    """cell 周圍曼哈頓距離 <= radius、在地圖內且未知的格子數。"""
    r0, c0 = cell
    count = 0
    for r in range(r0 - radius, r0 + radius + 1):
        for c in range(c0 - radius, c0 + radius + 1):
            if (
                abs(r - r0) + abs(c - c0) <= radius
                and 0 <= r < agent.rows
                and 0 <= c < agent.cols
                and (r, c) not in agent.known
            ):
                count += 1
    return count


def make_variant(name, cost_fn):
    class Variant(Agent):
        def _frontier_cost(self, cell, distance):
            return cost_fn(self, cell, distance)

    Variant.__name__ = name
    return Variant


def far_from_start(agent, cell):
    return abs(cell[0] - agent.start_position[0]) + abs(cell[1] - agent.start_position[1])


def combo(a_coef, radius, b_coef):
    return lambda a, f, d: d - a_coef * unknown_within(a, f, radius) - b_coef * far_from_start(a, f)


VARIANTS = {
    "base": lambda a, f, d: d,
    # 第一輪：單一想法
    "gain a=0.5": lambda a, f, d: d - 0.5 * unknown_within(a, f, 1),
    "area r=2 a=0.3": lambda a, f, d: d - 0.3 * unknown_within(a, f, 2),
    "area r=3 a=0.2": lambda a, f, d: d - 0.2 * unknown_within(a, f, 3),
    "far b=0.5": lambda a, f, d: d - 0.5 * far_from_start(a, f),
    # 第二輪：先驗加大、以及組合
    "far b=1.0": lambda a, f, d: d - 1.0 * far_from_start(a, f),
    "far b=1.5": lambda a, f, d: d - 1.5 * far_from_start(a, f),
    "r2 a.3 + b.5": combo(0.3, 2, 0.5),
    "r2 a.3 + b1": combo(0.3, 2, 1.0),
    "r2 a.5 + b.5": combo(0.5, 2, 0.5),
    "r3 a.2 + b.5": combo(0.2, 3, 0.5),
    "r3 a.3 + b.5": combo(0.3, 3, 0.5),
}


def main():
    dirs = [Path(p) for p in sys.argv[1:]] or [HW2_DIR / "maps"]
    print(f"{'variant':<18}" + "".join(f"{d.name:>10}" for d in dirs))
    for name, fn in VARIANTS.items():
        cls = make_variant(name, fn)
        cells = []
        for d in dirs:
            results = evaluate_dir(d, cls())
            fails = sum(not r["success"] for r in results)
            avg = sum(r["score"] for r in results) / len(results)
            cells.append(f"{avg:>8.2f}{'!' if fails else ' ':>2}")
        print(f"{name:<18}" + "".join(cells))


if __name__ == "__main__":
    main()
