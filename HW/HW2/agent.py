"""Starter code for the Unknown-Map Exploration Agent assignment."""

from collections import deque

MOVES = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}

class Agent:
    def __init__(self):
        """
        Initialize the Agent.

        You can declare any required instance variables here,
        but the state for each new map should be reinitialized in reset().
        """
        self.reset(0, 0, (0, 0))

    def reset(self, rows, cols, start_position):
        """
        Called once before each new map.

        Args:
            rows: 地圖列數
            cols: 地圖欄數
            start_position: 起點座標 (row, col)
        """
        self.rows = rows
        self.cols = cols
        self.start_position = tuple(start_position)
        # 已知地圖：(row, col) -> "FREE" / "WALL" / "GOAL"；不在 dict 裡就是未知
        self.known = {self.start_position: "FREE"}
        # 實際站過的格子
        self.visited = set()

    def act(self, percept):
        """
        Choose and return one of: UP, DOWN, LEFT, RIGHT.

        Args:
            percept:
                {
                    "position": (row, col),
                    "neighbors": {
                        "UP": "FREE" | "WALL" | "GOAL",
                        "DOWN": "FREE" | "WALL" | "GOAL",
                        "LEFT": "FREE" | "WALL" | "GOAL",
                        "RIGHT": "FREE" | "WALL" | "GOAL",
                    },
                }

        Returns:
            One of: UP, DOWN, LEFT, RIGHT
        """
        position = self._observe(percept)
        neighbors = percept["neighbors"]

        # 規則一：目標就在旁邊，直接走進去
        for action, status in neighbors.items():
            if status == "GOAL":
                return action

        # 規則二：在已知 FREE 格子上 BFS，走向最近的 frontier（只走已知 FREE，所以不會撞牆）
        first_action = {}
        queue = deque()
        for action in MOVES:
            nxt = self._step(position, action)
            if self.known.get(nxt) == "FREE":
                first_action[nxt] = action
                queue.append(nxt)
        while queue:
            cell = queue.popleft()
            if self._is_frontier(cell):
                return first_action[cell]
            for action in MOVES:
                nxt = self._step(cell, action)
                if self.known.get(nxt) == "FREE" and nxt not in first_action and nxt != position:
                    first_action[nxt] = first_action[cell]
                    queue.append(nxt)

        # 理論上不會發生（目標一定可達）；保險起見回傳一個不撞牆的方向
        for action, status in neighbors.items():
            if status == "FREE":
                return action
        return "UP"

    def _is_frontier(self, cell):
        """已知 FREE、而且還有地圖內的未知鄰居：站上去才會看到新東西。"""
        for action in MOVES:
            r, c = self._step(cell, action)
            if 0 <= r < self.rows and 0 <= c < self.cols and (r, c) not in self.known:
                return True
        return False

    def _observe(self, percept):
        """用感知更新已知地圖，回傳目前位置。"""
        position = tuple(percept["position"])
        self.known[position] = "FREE"
        self.visited.add(position)
        for action, status in percept["neighbors"].items():
            self.known[self._step(position, action)] = status
        return position

    @staticmethod
    def _step(position, action):
        dr, dc = MOVES[action]
        return (position[0] + dr, position[1] + dc)