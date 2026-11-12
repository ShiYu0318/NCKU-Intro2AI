"""HW4 Task 4：五名外送員，速率不變但會轉彎（類似在路口轉彎），每人有兩段 2–5 秒的 GPS 中斷。

完成 Tracker 並自行建立 P0、Q、R，再用 `python eval.py --task 4` 測試。
此檔案必須能單獨執行：只能 import numpy / filterpy / Python 標準函式庫，不可 import 其他 taskN.py。
"""
import numpy as np
from filterpy.kalman import KalmanFilter

# ===== 等速度模型（已給定，請勿修改）=====
# 狀態 x = [x, y, vx, vy]，單位 m 與 m/s；每一步 dt = 1 秒；GPS 只量到位置 (x, y)。
F = np.array([[1., 0., 1., 0.],
              [0., 1., 0., 1.],
              [0., 0., 1., 0.],
              [0., 0., 0., 1.]])
H = np.array([[1., 0., 0., 0.],
              [0., 1., 0., 0.]])


# Task 4：速率固定、只有方向會改變 → 加速度主要垂直於行進方向（法向大於切向）
Q_ALONG = 0.43     # 沿行進方向（切向）的加速度變異數
Q_ACROSS = 1.0   # 垂直行進方向（法向）的加速度變異數
Q_POS = 0.003      # 額外的位置隨機擾動變異數
# 白雜訊加速度模型（dt = 1）：加速度 a 對狀態的影響為 G @ a
G = np.array([[0.5, 0.], [0., 0.5], [1., 0.], [0., 1.]])


def make_matrices():
    """請自行建立並調整"""
    # 初始位置由 GPS 決定（不確定度 ≈ GPS 雜訊 5 m）；初速設為 0，其不確定度約為真實速度的變異
    P0 = np.diag([25., 25., 9.8, 9.8])
    # 基準 Q：方向未知時使用等向的加速度雜訊
    Q = G @ (np.eye(2) * (Q_ALONG + Q_ACROSS) / 2) @ G.T + Q_POS * np.diag([1., 1., 0., 0.])
    # GPS 每軸標準差約 5 m
    R = np.eye(2) * 25.
    return P0, Q, R


def directional_Q(velocity, base_Q):
    """依目前估計的行進方向，把加速度雜訊拆成切向 / 法向兩個分量。"""
    speed = float(np.hypot(velocity[0], velocity[1]))
    if speed < 0.5:  # 速度太小、方向不可靠時退回基準 Q
        return base_Q
    t = np.array([velocity[0], velocity[1]]) / speed  # 切向單位向量
    n = np.array([-t[1], t[0]])                        # 法向單位向量
    A = Q_ALONG * np.outer(t, t) + Q_ACROSS * np.outer(n, n)
    return G @ A @ G.T + Q_POS * np.diag([1., 1., 0., 0.])


class Tracker:
    def __init__(self, driver_ids):
        """driver_ids：這個 Scenario 的外送員 ID，例如 [1, 2, 3, 4, 5]。每個 Scenario 都會建立新的 Tracker。"""
        self.driver_ids = list(driver_ids)
        self.P0, self.Q, self.R = make_matrices()
        self.filters = {}

        # 為每位外送員建立 Kalman Filter
        for did in self.driver_ids:
            kf = KalmanFilter(dim_x=4, dim_z=2)
            kf.F = F.copy()
            kf.H = H.copy()
            kf.P = self.P0.copy()
            kf.Q = self.Q.copy()
            kf.R = self.R.copy()
            self.filters[did] = kf
        self.initialized = {did: False for did in self.driver_ids}

    def step(self, observations):
        """
        每秒由 eval.py 呼叫一次，估計所有外送員目前的位置。

        參數：
            observations：當前這一秒的 GPS 量測資料。
            格式：{driver_id: (gps_x, gps_y) 或 None}

            例如：
            {
                1: (12.3, -4.5),  # 外送員 1 有 GPS
                2: None           # 外送員 2 的 GPS 中斷
            }

        回傳：
            所有外送員的估計位置。
            格式：{driver_id: (estimated_x, estimated_y)}

            即使 GPS 中斷，也必須回傳該外送員的估計位置。

        實作要求：
            1. 第一次呼叫（t=0）：
            使用 GPS 初始化各外送員的位置，初始速度設為 0。

            2. 後續每次呼叫（t>=1）：
            - 執行 Kalman Filter Prediction。
            - 若有 GPS，執行 Update。
            - 若 GPS 為 None，跳過 Update。

            3. 回傳所有外送員的估計位置。
        """
        estimates = {}
        for did in self.driver_ids:
            kf = self.filters[did]
            z = observations.get(did)
            if not self.initialized[did]:
                # 第一次收到 GPS：用 GPS 初始化位置，速度設為 0
                if z is not None:
                    kf.x = np.array([float(z[0]), float(z[1]), 0., 0.])
                    kf.P = self.P0.copy()
                    self.initialized[did] = True
            else:
                kf.Q = directional_Q(kf.x[2:4], self.Q)
                kf.predict()
                if z is not None:
                    kf.update(np.array([float(z[0]), float(z[1])]))
            estimates[did] = (float(kf.x[0]), float(kf.x[1]))
        return estimates
