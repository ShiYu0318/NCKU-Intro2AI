"""HW4 本機測試工具：執行你的 taskN.py，並用公開 Ground Truth 計算 RMSE。

用法（在這個資料夾裡執行）：
    python eval.py --task 1                    # Task 1，Train + Validation
    python eval.py --task all                  # 四題全部
    python eval.py --task 3 --split val        # 只跑 Validation
    python eval.py --task 2 --save pred.csv    # 另存每秒預測

正式批改使用相同的介面檢查與 RMSE 計算，只是換成助教的 Hidden Test；
這裡的 RMSE 只是估計，正式分數以 Hidden Test 的 RMSE 換算。
"""
from pathlib import Path
import argparse
import csv
import importlib.util
import math
import numbers
import sys
import time

import numpy as np

DURATION_S = 60
WARMUP_S = 8  # t=0..7 照常執行 step()，但不計入 RMSE
RIDERS_PER_TASK = {1: 1, 2: 1, 3: 3, 4: 5}
HERE = Path(__file__).resolve().parent


class TrackerError(RuntimeError):
    """你的 Tracker 違反介面或執行失敗；正式批改時該題 0 分。"""


def read_scenarios(input_csv):
    """回傳 [(scenario_id, rider_ids, frames)]，frames[t] = {id: (x, y) 或 None}。"""
    scenarios = {}
    with Path(input_csv).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            sid, t, rid = row["scenario_id"], int(row["time_s"]), int(row["driver_id"])
            x, y = float(row["Measured_X"]), float(row["Measured_Y"])
            frames = scenarios.setdefault(sid, [{} for _ in range(DURATION_S)])
            frames[t][rid] = None if math.isnan(x) else (x, y)
    result = []
    for sid, frames in scenarios.items():
        rider_ids = sorted(frames[0])
        if any(sorted(frame) != rider_ids for frame in frames):
            raise ValueError(f"{sid}: incomplete time/ID grid in {input_csv}")
        result.append((sid, rider_ids, frames))
    return result


def read_truth(truth_csv):
    with Path(truth_csv).open(encoding="utf-8-sig", newline="") as handle:
        return {(r["scenario_id"], int(r["time_s"]), int(r["driver_id"])): (float(r["true_x"]), float(r["true_y"]))
                for r in csv.DictReader(handle)}


def load_tracker_class(path):
    path = Path(path).resolve(strict=True)
    spec = importlib.util.spec_from_file_location(f"hw4_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = getattr(module, "Tracker", None)
    if not isinstance(cls, type) or not callable(getattr(cls, "step", None)):
        raise TrackerError(f"{path.name} 必須定義 class Tracker，且有 step() 方法")
    return cls


def check_output(output, rider_ids, sid, t):
    """每秒必須回傳 {每個 ID: (x, y)}，座標為有限實數。"""
    where = f"{sid} t={t}"
    if not isinstance(output, dict):
        raise TrackerError(f"{where}: step() 必須回傳 dict，收到 {type(output).__name__}")
    if sorted(output, key=repr) != sorted(rider_ids, key=repr):
        raise TrackerError(f"{where}: 回傳的 ID {sorted(output, key=repr)} 必須正好是 {rider_ids}")
    clean = {}
    for rid in rider_ids:
        value = output[rid]
        try:
            if isinstance(value, (str, bytes)) or len(value) != 2:
                raise TypeError
            if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real) for v in value):
                raise TypeError
            x, y = (float(v) for v in value)
        except (TypeError, ValueError):
            raise TrackerError(f"{where}: ID {rid} 必須回傳兩個數字 (x, y)，收到 {value!r}") from None
        if not (math.isfinite(x) and math.isfinite(y)):
            raise TrackerError(f"{where}: ID {rid} 回傳了 NaN 或 Infinity")
        clean[rid] = (x, y)
    return clean


def run_tracker(tracker_cls, scenarios):
    """每個 Scenario 建立新的 Tracker，逐秒只傳入當下的 GPS。回傳 {(sid, t, id): (x, y)}。"""
    predictions = {}
    for sid, rider_ids, frames in scenarios:
        try:
            tracker = tracker_cls(list(rider_ids))
        except (Exception, SystemExit) as exc:
            raise TrackerError(f"{sid}: Tracker(rider_ids) 失敗：{type(exc).__name__}: {exc}") from exc
        for t, frame in enumerate(frames):
            observations = {rid: frame[rid] for rid in rider_ids}  # 每秒新的 dict
            try:
                output = tracker.step(observations)
            except (Exception, SystemExit) as exc:
                raise TrackerError(f"{sid} t={t}: step() 失敗：{type(exc).__name__}: {exc}") from exc
            for rid, xy in check_output(output, rider_ids, sid, t).items():
                predictions[(sid, t, rid)] = xy
    return predictions


def rmse(predictions, truth):
    keys = [k for k in truth if k[1] >= WARMUP_S]
    sse = math.fsum((predictions[k][0] - truth[k][0]) ** 2 + (predictions[k][1] - truth[k][1]) ** 2 for k in keys)
    if not math.isfinite(sse):
        raise TrackerError("預測值過大，平方誤差溢位")
    return math.sqrt(sse / len(keys))


def save_predictions(predictions, path):
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario_id", "time_s", "driver_id", "Estimated_X", "Estimated_Y"])
        for (sid, t, rid), (x, y) in predictions.items():
            writer.writerow([sid, t, rid, x, y])


def main():
    parser = argparse.ArgumentParser(description="HW4 本機測試工具")
    parser.add_argument("--task", required=True, choices=["1", "2", "3", "4", "all"])
    parser.add_argument("--split", default="both", choices=["train", "val", "both"])
    parser.add_argument("--data", default=str(HERE / "data"), help="資料夾，預設 ./data")
    parser.add_argument("--code", default=str(HERE), help="taskN.py 所在資料夾，預設本資料夾")
    parser.add_argument("--save", help="另存預測 CSV（只能搭配單一題與單一 split）")
    args = parser.parse_args()
    tasks = [1, 2, 3, 4] if args.task == "all" else [int(args.task)]
    splits = ["train", "val"] if args.split == "both" else [args.split]
    if args.save and (len(tasks) > 1 or len(splits) > 1):
        parser.error("--save 只能搭配單一 --task 與單一 --split")
    failed = False
    for task in tasks:
        for split in splits:
            folder = Path(args.data) / f"task{task}"
            label = f"Task {task} {split:<5}"
            try:
                started = time.perf_counter()
                tracker_cls = load_tracker_class(Path(args.code) / f"task{task}.py")
                predictions = run_tracker(tracker_cls, read_scenarios(folder / f"{split}.csv"))
                value = rmse(predictions, read_truth(folder / f"{split}_gt.csv"))
                print(f"{label} RMSE = {value:6.3f} m   ({time.perf_counter() - started:.1f}s)")
                if args.save:
                    save_predictions(predictions, args.save)
                    print(f"已儲存：{args.save}")
            except (Exception, SystemExit) as exc:
                failed = True
                print(f"{label} 錯誤：{type(exc).__name__}: {exc}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
