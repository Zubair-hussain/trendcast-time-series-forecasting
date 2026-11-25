"""
TrendCast core model.

An interpretable ensemble for short-horizon trend forecasting of a univariate
engagement signal (views, searches, plays, mentions ...). It combines:

  1. Holt double-exponential smoothing  -> captures level + local slope
  2. Ridge regression on lag/rolling features -> captures short memory
  3. A momentum ("virality") score in [0, 100] derived from normalised
     acceleration, so a product UI can label a series as rising/steady/fading.

The whole thing is pure NumPy + scikit-learn, so it trains in milliseconds and
runs on any laptop with no GPU. Everything here is written from scratch for this
project; it is not a wrapper around a third-party model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.linear_model import Ridge


@dataclass
class TrendForecast:
    """Result of a forecast call."""

    horizon: int
    point: np.ndarray            # predicted values, shape (horizon,)
    lower: np.ndarray            # lower band (approx. 80% interval)
    upper: np.ndarray            # upper band
    momentum: float              # 0..100 virality/momentum score
    label: str                   # "rising" | "steady" | "fading"

    def as_dict(self) -> dict:
        return {
            "horizon": self.horizon,
            "point": [round(float(v), 4) for v in self.point],
            "lower": [round(float(v), 4) for v in self.lower],
            "upper": [round(float(v), 4) for v in self.upper],
            "momentum": round(float(self.momentum), 2),
            "label": self.label,
        }


def _holt(series: np.ndarray, alpha: float, beta: float):
    """Holt double-exponential smoothing. Returns (level, trend) arrays."""
    n = len(series)
    level = np.zeros(n)
    trend = np.zeros(n)
    level[0] = series[0]
    trend[0] = series[1] - series[0] if n > 1 else 0.0
    for t in range(1, n):
        prev = level[t - 1]
        level[t] = alpha * series[t] + (1 - alpha) * (prev + trend[t - 1])
        trend[t] = beta * (level[t] - prev) + (1 - beta) * trend[t - 1]
    return level, trend


def _make_features(series: np.ndarray, lags: int, roll: int) -> np.ndarray:
    """Lag + rolling-mean + rolling-std features for the ridge stage."""
    n = len(series)
    rows = []
    for t in range(n):
        feat = []
        for l in range(1, lags + 1):
            feat.append(series[t - l] if t - l >= 0 else series[0])
        window = series[max(0, t - roll): t] if t > 0 else series[:1]
        feat.append(float(np.mean(window)))
        feat.append(float(np.std(window)))
        rows.append(feat)
    return np.asarray(rows, dtype=float)


class TrendCast:
    """Fit on one historical series, then forecast the next `horizon` steps."""

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        lags: int = 7,
        roll: int = 7,
        ridge_alpha: float = 1.0,
        ensemble_weight: float = 0.3,
    ):
        # ensemble_weight = weight on Holt vs ridge (0..1)
        self.alpha = alpha
        self.beta = beta
        self.lags = lags
        self.roll = roll
        self.ridge_alpha = ridge_alpha
        self.ensemble_weight = ensemble_weight
        self._ridge: Ridge | None = None
        self._series: np.ndarray | None = None
        self._resid_std: float = 0.0

    def fit(self, series: Sequence[float]) -> "TrendCast":
        s = np.asarray(series, dtype=float)
        if s.ndim != 1 or len(s) < max(self.lags + 1, 3):
            raise ValueError("series must be 1-D with at least lags+1 points")
        self._series = s

        X = _make_features(s, self.lags, self.roll)
        y = s
        self._ridge = Ridge(alpha=self.ridge_alpha)
        self._ridge.fit(X[1:], y[1:])          # predict next from past

        # in-sample residual std for interval bands
        fitted = self._ridge.predict(X[1:])
        self._resid_std = float(np.std(y[1:] - fitted)) or 1e-6
        return self

    def _momentum(self, s: np.ndarray) -> tuple[float, str]:
        """Normalised recent acceleration -> 0..100 score + label."""
        if len(s) < 4:
            return 50.0, "steady"
        recent = s[-min(len(s), 2 * self.roll):]
        slope = np.polyfit(np.arange(len(recent)), recent, 1)[0]
        # scale by local step volatility so a clean trend scores decisively
        scale = np.std(np.diff(recent)) + 1e-6
        z = float(np.clip(slope / scale, -20, 20))
        score = float(100.0 / (1.0 + np.exp(-1.5 * z)))   # logistic squash
        if score >= 60:
            label = "rising"
        elif score <= 40:
            label = "fading"
        else:
            label = "steady"
        return score, label

    def forecast(self, horizon: int = 7) -> TrendForecast:
        if self._ridge is None or self._series is None:
            raise RuntimeError("call fit() before forecast()")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")

        s = self._series.copy()
        level, trend = _holt(s, self.alpha, self.beta)
        last_level, last_trend = level[-1], trend[-1]

        w = self.ensemble_weight
        preds = []
        work = s.copy()
        for h in range(1, horizon + 1):
            holt_pred = last_level + h * last_trend
            feat = _make_features(np.append(work, work[-1]), self.lags, self.roll)[-1]
            ridge_pred = float(self._ridge.predict(feat.reshape(1, -1))[0])
            p = w * holt_pred + (1 - w) * ridge_pred
            preds.append(p)
            work = np.append(work, p)

        point = np.asarray(preds)
        # bands widen with sqrt(horizon)
        spread = 1.28 * self._resid_std * np.sqrt(np.arange(1, horizon + 1))
        momentum, label = self._momentum(s)
        return TrendForecast(
            horizon=horizon,
            point=point,
            lower=point - spread,
            upper=point + spread,
            momentum=momentum,
            label=label,
        )


def mape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean absolute percentage error (%)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(yt) < 1e-9, 1e-9, np.abs(yt))
    return float(np.mean(np.abs((yt - yp) / denom)) * 100.0)


def smape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Symmetric MAPE (%)."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    denom = (np.abs(yt) + np.abs(yp)) / 2.0 + 1e-9
    return float(np.mean(np.abs(yt - yp) / denom) * 100.0)
