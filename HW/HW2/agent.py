"""Starter code for the Unknown-Map Exploration Agent assignment."""

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
        # DFS 的回溯路徑：從起點到目前位置的上一格
        self.stack = []

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

        # 規則二：只考慮 FREE 的方向，永遠不撞牆
        for action, status in neighbors.items():
            if status == "FREE" and self._step(position, action) not in self.visited:
                # 往前探索：記住來時的格子，之後才能退回來
                self.stack.append(position)
                return action

        # 死路：沿原路退一格
        parent = self.stack.pop()
        return self._direction(position, parent)

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

    @staticmethod
    def _direction(src, dst):
        """相鄰兩格之間的移動方向。"""
        delta = (dst[0] - src[0], dst[1] - src[1])
        for action, move in MOVES.items():
            if move == delta:
                return action
        raise ValueError(f"{src} and {dst} are not adjacent")
