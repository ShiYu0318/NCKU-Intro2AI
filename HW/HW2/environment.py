"""Public grid-world environment. Students should not modify this file."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


ACTIONS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}


@dataclass(frozen=True)
class MapDefinition:
    name: str
    rows: int
    cols: int
    grid: tuple[str, ...]
    start: tuple[int, int]
    goal: tuple[int, int]
    style: str = "unspecified"

    @classmethod
    def from_json(cls, path: str | Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        result = cls(
            name=data["name"],
            rows=int(data["rows"]),
            cols=int(data["cols"]),
            grid=tuple(data["grid"]),
            start=tuple(data["start"]),
            goal=tuple(data["goal"]),
            style=data.get("style", "unspecified"),
        )
        result.validate()
        return result

    def validate(self):
        if len(self.grid) != self.rows or any(len(row) != self.cols for row in self.grid):
            raise ValueError(f"{self.name}: grid dimensions do not match rows/cols")
        for label, cell in (("start", self.start), ("goal", self.goal)):
            r, c = cell
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise ValueError(f"{self.name}: {label} is outside the map")
            if self.grid[r][c] == "#":
                raise ValueError(f"{self.name}: {label} is inside a wall")
        if self.start == self.goal:
            raise ValueError(f"{self.name}: start and goal must differ")


class GridEnvironment:
    def __init__(self, definition: MapDefinition):
        self.definition = definition
        self.position = definition.start
        self.steps = 0
        self.collisions = 0
        self.done = False
        self.max_steps = 4 * definition.rows * definition.cols

    def _cell_status(self, row: int, col: int):
        if not (0 <= row < self.definition.rows and 0 <= col < self.definition.cols):
            return "WALL"
        if self.definition.grid[row][col] == "#":
            return "WALL"
        if (row, col) == self.definition.goal:
            return "GOAL"
        return "FREE"

    def observe(self):
        row, col = self.position
        return {
            "position": self.position,
            "neighbors": {
                action: self._cell_status(row + dr, col + dc)
                for action, (dr, dc) in ACTIONS.items()
            },
        }

    def step(self, action: str):
        if self.done:
            raise RuntimeError("The episode has already ended")
        if action not in ACTIONS:
            raise ValueError(f"Invalid action: {action!r}")

        self.steps += 1
        row, col = self.position
        dr, dc = ACTIONS[action]
        target = (row + dr, col + dc)
        status = self._cell_status(*target)
        if status == "WALL":
            self.collisions += 1
        else:
            self.position = target
            if target == self.definition.goal:
                self.done = True
        return self.done

    @property
    def exhausted(self):
        return self.steps >= self.max_steps and not self.done
