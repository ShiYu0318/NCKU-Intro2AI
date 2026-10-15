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
        free = [a for a, s in neighbors.items() if s == "FREE"]
        unvisited = [a for a in free if self._step(position, a) not in self.visited]
        if unvisited:
            return unvisited[0]
        return free[0]

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
