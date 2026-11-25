"""
Benchmark TrendCast against two standard baselines on the synthetic dataset:

  * naive     — tomorrow = today (persistence)
  * drift     — linear extrapolation of the last `roll` points
  * trendcast — this project's ensemble

Reports sMAPE / MAPE (lower is better) and fit+predict latency.
Run:  python benchmarks/benchmark.py
"""

from __future__ import annotations

import time

import numpy as np

from trendcast.dataset import generate_dataset
from trendcast.model import TrendCast, smape, mape


def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def naive_forecast(train, h):
    return np.repeat(train[-1], h)


def drift_forecast(train, h, roll=7):
    w = train[-roll:]
    slope = np.polyfit(np.arange(len(w)), w, 1)[0]
    return train[-1] + slope * np.arange(1, h + 1)


def run(horizon: int = 14):
    data = generate_dataset()
    rows = {"naive": [], "drift": [], "trendcast": []}
    smape_rows = {k: [] for k in rows}
    mae_rows = {k: [] for k in rows}
    t0 = time.perf_counter()
    n_fore = 0

    for s in data:
        series = s.values
        if len(series) <= horizon + 10:
            continue
        train, actual = series[:-horizon], series[-horizon:]

        for name, pred in (
            ("naive", naive_forecast(train, horizon)),
            ("drift", drift_forecast(train, horizon)),
            ("trendcast", TrendCast().fit(train).forecast(horizon).point),
        ):
            rows[name].append(mape(actual, pred))
            smape_rows[name].append(smape(actual, pred))
            mae_rows[name].append(mae(actual, pred))
        n_fore += 1

    elapsed = time.perf_counter() - t0

    print(f"\nTrendCast benchmark — {n_fore} series, horizon={horizon}\n")
    print(f"{'model':<12}{'sMAPE %':>10}{'MAE':>10}")
    print("-" * 32)
    for name in rows:
        print(f"{name:<12}{np.mean(smape_rows[name]):>10.2f}{np.mean(mae_rows[name]):>10.2f}")
    print("-" * 32)
    print(f"total wall time: {elapsed*1000:.0f} ms "
          f"({elapsed/max(n_fore,1)*1000:.2f} ms / series, fit+forecast)\n")


if __name__ == "__main__":
    run()
