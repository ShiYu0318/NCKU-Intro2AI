"""
Unknown-Map Exploration Agent：frontier-based exploration。

每一步：
1. 把感知到的四個鄰居寫進自己的已知地圖。
2. 目標在旁邊就直接走進去。
3. 否則在「已知可走」的格子上做 BFS，找出所有 frontier
   （已知可走、但還有未知鄰居的格子），挑成本最低的那個，回傳通往它的第一步。

只走已知可走的格子，所以碰撞次數永遠是 0。
frontier 成本 = 距離 − 資訊增益 − 離起點的遠近（目標通常放在離起點很遠的地方）。
"""

from collections import deque

MOVES = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}

class Agent:
    # frontier 評分參數（由 tools/experiment.py 在隨機地圖上實驗選出）
    INFO_RADIUS = 2    # 計算資訊增益時看的曼哈頓半徑
    INFO_WEIGHT = 0.4  # 周圍每多一個未知格，成本少多少
    FAR_WEIGHT = 0.75  # 離起點每遠一格，成本少多少（目標通常離起點很遠）

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

        # 規則二：在已知 FREE 格子上 BFS，走向成本最低的 frontier（只走已知 FREE，所以不會撞牆）
        dist, first_action = self._bfs(position)
        frontiers = [cell for cell in dist if self._is_frontier(cell)]
        if frontiers:
            target = min(frontiers, key=lambda cell: self._frontier_cost(cell, dist[cell]))
            return first_action[target]

        # 理論上不會發生（目標一定可達）；保險起見回傳一個不撞牆的方向
        for action, status in neighbors.items():
            if status == "FREE":
                return action
        return "UP"

    def _frontier_cost(self, cell, distance):
        """
        frontier 的成本，越低越優先：
            走過去的步數
            - INFO_WEIGHT × 周圍的未知格數（資訊增益：去了能看到更多）
            - FAR_WEIGHT × 離起點的曼哈頓距離（先驗：目標通常在遠處）
        """
        far = abs(cell[0] - self.start_position[0]) + abs(cell[1] - self.start_position[1])
        return distance - self.INFO_WEIGHT * self._unknown_near(cell) - self.FAR_WEIGHT * far

    def _unknown_near(self, cell):
        """cell 周圍曼哈頓距離 <= INFO_RADIUS、在地圖內且未知的格子數。"""
        r0, c0 = cell
        radius = self.INFO_RADIUS
        count = 0
        for r in range(max(0, r0 - radius), min(self.rows, r0 + radius + 1)):
            span = radius - abs(r - r0)
            for c in range(max(0, c0 - span), min(self.cols, c0 + span + 1)):
                if (r, c) not in self.known:
                    count += 1
        return count

    def _bfs(self, position):
        """
        從目前位置在已知 FREE 格子上做 BFS。

        Returns:
            dist: 格子 -> 最短步數（不含目前位置，依 BFS 順序插入）
            first_action: 格子 -> 走向它時，從目前位置出發的第一步
        """
        dist = {}
        first_action = {}
        queue = deque()
        for action in MOVES:
            nxt = self._step(position, action)
            if self.known.get(nxt) == "FREE":
                dist[nxt] = 1
                first_action[nxt] = action
                queue.append(nxt)
        while queue:
            cell = queue.popleft()
            for action in MOVES:
                nxt = self._step(cell, action)
                if self.known.get(nxt) == "FREE" and nxt not in dist and nxt != position:
                    dist[nxt] = dist[cell] + 1
                    first_action[nxt] = first_action[cell]
                    queue.append(nxt)
        return dist, first_action

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
        for action, status in percept["neighbors"].items():
            self.known[self._step(position, action)] = status
        return position

    @staticmethod
    def _step(position, action):
        dr, dc = MOVES[action]
        return (position[0] + dr, position[1] + dc)