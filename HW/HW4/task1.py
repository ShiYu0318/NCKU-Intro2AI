"""HW4 Task 1：單一外送員直線等速行駛，GPS 不會缺失。

完成 Tracker 並自行建立 P0、Q、R，再用 `python eval.py --task 1` 測試。
此檔案必須能單獨執行：只能 import numpy / filterpy / Python 標準函式庫，不可 import 其他 taskN.py。
"""
import numpy as np
from filterpy.kalman import KalmanFilter

# ===== 等速度模型（已給定，請勿修改）=====
F = np.array([[1., 0., 1., 0.],
              [0., 1., 0., 1.],
              [0., 0., 1., 0.],
              [0., 0., 0., 1.]])
H = np.array([[1., 0., 0., 0.],
              [0., 1., 0., 0.]])


def make_matrices():
    """請自行建立並調整"""
    # 初始位置由 GPS 決定（不確定度約等於 GPS 雜訊 5 m），初速設為 0，不確定度較大
    P0 = np.diag([25., 25., 9.0, 9.0])
    # 離散白雜訊加速度模型（dt = 1）：加速度變異數 q
    q = 1e-05
    G = np.array([[0.5, 0.], [0., 0.5], [1., 0.], [0., 1.]])
    Q = q * G @ G.T
    # GPS 每軸標準差約 5 m
    R = np.eye(2) * 25.
    return P0, Q, R


class Tracker:
    def __init__(self, driver_ids):
        """driver_ids：這個 Scenario 的外送員 ID，例如 [1]。每個 Scenario 都會建立新的 Tracker。"""
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
            格式：{driver_id: (gps_x, gps_y)}

            例如：
            {1: (12.3, -4.5)}

        回傳：
            所有外送員的估計位置。
            格式：{driver_id: (estimated_x, estimated_y)}

        實作要求：
            1. 第一次呼叫（t=0）：
            使用 GPS 初始化各外送員的位置，初始速度設為 0。

            2. 後續每次呼叫（t>=1）：
            - 執行 Kalman Filter Prediction。
            - 使用 GPS 執行 Update。

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
                kf.predict()
                if z is not None:
                    kf.update(np.array([float(z[0]), float(z[1])]))
            estimates[did] = (float(kf.x[0]), float(kf.x[1]))
        return estimates

