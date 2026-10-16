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
    # 第三輪：very_far 地圖上，先驗再加重
    "r2 a.5 + b1": combo(0.5, 2, 1.0),
    "r2 a.5 + b2": combo(0.5, 2, 2.0),
    "r2 a.5 + b3": combo(0.5, 2, 3.0),
    # 第四輪：先驗改用已知路徑距離
    "manhattan a.4 b.75": combo(0.4, 2, 0.75),
}


class PathPrior(Agent):
    """先驗改用「從起點沿已知 FREE 格子走到 frontier 的 BFS 距離」。"""

    a_coef, radius, b_coef, mix = 0.4, 2, 0.75, 1.0  # mix=1 純路徑距離，0 純曼哈頓

    def act(self, percept):
        self._observe(percept)
        self._start_dist, _ = self._bfs(self.start_position)
        self._start_dist[self.start_position] = 0
        return super().act(percept)

    def _frontier_cost(self, cell, distance):
        prior = self.mix * self._start_dist.get(cell, 0) + (1 - self.mix) * far_from_start(self, cell)
        return distance - self.a_coef * unknown_within(self, cell, self.radius) - self.b_coef * prior


def path_prior(a_coef, radius, b_coef, mix):
    return type("PathPrior", (PathPrior,), dict(a_coef=a_coef, radius=radius, b_coef=b_coef, mix=mix))


VARIANTS.update({
    "path a.4 b.75": path_prior(0.4, 2, 0.75, 1.0),
    "path a.4 b1": path_prior(0.4, 2, 1.0, 1.0),
    "mix.5 a.4 b.75": path_prior(0.4, 2, 0.75, 0.5),
    "mix.5 a.4 b1": path_prior(0.4, 2, 1.0, 0.5),
})


def main():
    # --only=子字串：只跑名稱包含該字串的變體（可用逗號分隔多個）
    only = [a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("--only=")]
    only = only[0] if only else None
    dirs = [Path(p) for p in sys.argv[1:] if not p.startswith("--")] or [HW2_DIR / "maps"]
    print(f"{'variant':<18}" + "".join(f"{d.name:>10}" for d in dirs))
    for name, fn in VARIANTS.items():
        if only and not any(key in name for key in only):
            continue
        cls = fn if isinstance(fn, type) else make_variant(name, fn)
        cells = []
        for d in dirs:
            results = evaluate_dir(d, cls())
            fails = sum(not r["success"] for r in results)
            avg = sum(r["score"] for r in results) / len(results)
            cells.append(f"{avg:>8.2f}{'!' if fails else ' ':>2}")
        print(f"{name:<18}" + "".join(cells))


if __name__ == "__main__":
    main()
