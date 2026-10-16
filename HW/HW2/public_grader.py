"""Run a student's agent on the three published maps."""

from __future__ import annotations

import argparse
from pathlib import Path

from agent import Agent
from environment import ACTIONS, GridEnvironment, MapDefinition


def evaluate(agent, path: Path, verbose: bool):
    definition = MapDefinition.from_json(path)
    env = GridEnvironment(definition)
    agent.reset(definition.rows, definition.cols, definition.start)

    while not env.done and not env.exhausted:
        percept = env.observe()
        action = agent.act(percept)
        if action not in ACTIONS:
            raise ValueError(f"Agent returned invalid action {action!r}")
        env.step(action)
        if verbose:
            print(
                f"  step={env.steps:4d}  position={str(env.position):>9}  "
                f"action={action:<5}  collisions={env.collisions}"
            )

    return {
        "map": definition.name,
        "success": env.done,
        "steps": env.steps,
        "collisions": env.collisions,
    }


def print_banner():
    width = 66
    print("\n" + "=" * width)
    print("HW2 - PUBLIC TEST".center(width))
    print("=" * width)


def print_results_table(results):
    headers = ("Map", "Result", "Steps", "Collisions")
    widths = (22, 10, 10, 12)

    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

    def row(values):
        return " | ".join(
            str(value).ljust(width)
            for value, width in zip(values, widths)
        )

    separator = "-+-".join("-" * width for width in widths)
    print("\n" + row(headers))
    print(separator)

    for result in results:
        if "error" in result:
            status = "ERROR"
            color = RED
            values = (result["map"], status, "-", "-")
        else:
            status = "SUCCESS" if result["success"] else "FAILED"
            color = GREEN if result["success"] else RED
            values = (
                result["map"],
                status,
                result["steps"],
                result["collisions"],
            )

        line = row(values)

        # 先排版完，再只替換 Result 文字的顏色
        line = line.replace(status, f"{color}{status}{RESET}", 1)
        print(line)

        if "error" in result:
            print(f"  Error: {result['error']}")

    print(separator)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    maps = sorted((Path(__file__).parent / "maps").glob("*.json"))
    if not maps:
        raise SystemExit("No public maps were found in the maps/ directory.")

    print_banner()
    agent = Agent()
    results = []

    for path in maps:
        try:
            result = evaluate(agent, path, args.verbose)
        except Exception as exc:
            result = {
                "map": path.stem,
                "success": False,
                "steps": 0,
                "collisions": 0,
                "error": str(exc),
            }
        results.append(result)

    print_results_table(results)
    passed = sum(result.get("success", False) for result in results)
    total_steps = sum(result.get("steps", 0) for result in results)
    total_collisions = sum(result.get("collisions", 0) for result in results)

    print(f"\nSuccessful maps : {passed}/{len(results)}")
    print("=" * 66 + "\n")


if __name__ == "__main__":
    main()
