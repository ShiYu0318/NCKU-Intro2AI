"""本機評分工具：跑 Agent 並依 README 9.2 的公式計算分數。

用法（在 HW2/ 底下）：
    python3 tools/local_score.py                 # 跑 maps/ 的公開地圖
    python3 tools/local_score.py some_dir/       # 跑指定資料夾裡的所有 .json 地圖

這支檔案只用來自我檢測，不需要繳交。它會讀取完整地圖來算最短路徑，
這是「評分者」的角色；agent.py 本身絕對不能這樣做。
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

HW2_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HW2_DIR))

from agent import Agent  # noqa: E402
from environment import ACTIONS, GridEnvironment, MapDefinition  # noqa: E402


def optimal_steps(definition: MapDefinition) -> int:
    """在完整地圖上用 BFS 算起點到目標的最短步數。"""
    start, goal = definition.start, definition.goal
    dist = {start: 0}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        if (r, c) == goal:
            return dist[(r, c)]
        for dr, dc in ACTIONS.values():
            nr, nc = r + dr, c + dc
            if (
                0 <= nr < definition.rows
                and 0 <= nc < definition.cols
                and definition.grid[nr][nc] != "#"
                and (nr, nc) not in dist
            ):
                dist[(nr, nc)] = dist[(r, c)] + 1
                queue.append((nr, nc))
    raise ValueError(f"{definition.name}: goal is unreachable")


def score_of(success: bool, steps: int, collisions: int, optimal: int) -> float:
    """README 9.2 的單張地圖分數公式。"""
    if not success:
        return 0.0
    movement = steps - collisions
    efficiency = min(1.0, optimal / movement)
    return 80 + 20 * efficiency - min(10, collisions)


def run_map(agent: Agent, definition: MapDefinition) -> dict:
    """和 public_grader.evaluate 相同的流程。"""
    env = GridEnvironment(definition)
    agent.reset(definition.rows, definition.cols, definition.start)
    while not env.done and not env.exhausted:
        action = agent.act(env.observe())
        if action not in ACTIONS:
            raise ValueError(f"Agent returned invalid action {action!r}")
        env.step(action)
    opt = optimal_steps(definition)
    return {
        "map": definition.name,
        "style": definition.style,
        "success": env.done,
        "steps": env.steps,
        "collisions": env.collisions,
        "optimal": opt,
        "score": score_of(env.done, env.steps, env.collisions, opt),
    }


def evaluate_dir(map_dir: Path, agent: Agent | None = None) -> list[dict]:
    agent = agent or Agent()  # 和 grader 一樣，同一個物件連續跑多張圖
    return [
        run_map(agent, MapDefinition.from_json(path))
        for path in sorted(map_dir.glob("*.json"))
    ]


def print_summary(results: list[dict]) -> None:
    """只印各 style / 尺寸的平均，適合大量地圖。"""
    groups: dict[str, list[dict]] = {}
    for r in results:
        size = r["map"].split("_")[-2] if r["map"].count("_") >= 2 else "?"
        groups.setdefault(f"{r['style']:<12} {size:>2}", []).append(r)
    for key in sorted(groups):
        rs = groups[key]
        avg = sum(r["score"] for r in rs) / len(rs)
        fail = sum(not r["success"] for r in rs)
        coll = sum(r["collisions"] for r in rs)
        ratio = sum(r["optimal"] / max(1, r["steps"] - r["collisions"]) for r in rs) / len(rs)
        print(f"{key}  n={len(rs):<4} avg={avg:6.2f}  eff={ratio:.3f}  fail={fail}  coll={coll}")
    avg = sum(r["score"] for r in results) / len(results)
    print(f"ALL               n={len(results):<4} avg={avg:6.2f}")


def print_report(results: list[dict]) -> None:
    header = f"{'map':<22}{'style':<14}{'ok':<4}{'steps':>6}{'coll':>6}{'opt':>5}{'score':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['map']:<22}{r['style']:<14}{'Y' if r['success'] else 'N':<4}"
            f"{r['steps']:>6}{r['collisions']:>6}{r['optimal']:>5}{r['score']:>8.2f}"
        )
    print("-" * len(header))
    avg = sum(r["score"] for r in results) / len(results)
    ok = sum(r["success"] for r in results)
    print(f"success {ok}/{len(results)}   average score {avg:.2f}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    target = Path(args[0]) if args else HW2_DIR / "maps"
    results = evaluate_dir(target)
    if "--summary" in sys.argv:
        print_summary(results)
    else:
        print_report(results)
