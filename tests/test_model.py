"""Unit + behavioural test cases for TrendCast."""

import numpy as np
import pytest

from trendcast.model import TrendCast, mape, smape
from trendcast.dataset import generate_dataset, save_csv, load_dataset


# ----- fixtures -------------------------------------------------------------
@pytest.fixture
def rising_series():
    t = np.arange(60)
    return 10 + 2.0 * t + np.random.default_rng(0).standard_normal(60)


@pytest.fixture
def fading_series():
    t = np.arange(60)
    return np.clip(130 - 2.0 * t, 1, None)


# ----- shape / contract -----------------------------------------------------
def test_forecast_shapes(rising_series):
    fc = TrendCast().fit(rising_series).forecast(horizon=7)
    assert fc.point.shape == (7,)
    assert fc.lower.shape == (7,) and fc.upper.shape == (7,)
    assert np.all(fc.upper >= fc.lower)


def test_requires_fit():
    with pytest.raises(RuntimeError):
        TrendCast().forecast(5)


def test_rejects_short_series():
    with pytest.raises(ValueError):
        TrendCast(lags=7).fit([1, 2, 3])


def test_rejects_bad_horizon(rising_series):
    with pytest.raises(ValueError):
        TrendCast().fit(rising_series).forecast(0)


# ----- behaviour ------------------------------------------------------------
def test_rising_label_and_momentum(rising_series):
    fc = TrendCast().fit(rising_series).forecast(7)
    assert fc.label == "rising"
    assert fc.momentum > 60


def test_fading_label(fading_series):
    fc = TrendCast().fit(fading_series).forecast(7)
    assert fc.label == "fading"
    assert fc.momentum < 40


def test_forecast_direction_up(rising_series):
    fc = TrendCast().fit(rising_series).forecast(7)
    assert fc.point[-1] > fc.point[0]


def test_accuracy_on_clean_trend(rising_series):
    train, actual = rising_series[:-7], rising_series[-7:]
    fc = TrendCast().fit(train).forecast(7)
    assert mape(actual, fc.point) < 15.0


# ----- dataset round-trip ---------------------------------------------------
def test_dataset_roundtrip(tmp_path):
    data = generate_dataset(n_series=4, length=40, seed=1)
    p = tmp_path / "d.csv"
    save_csv(data, str(p))
    loaded = load_dataset(str(p))
    assert len(loaded) == 4
    assert all(len(v) == 40 for v in loaded.values())


def test_dataset_deterministic():
    a = generate_dataset(n_series=3, length=30, seed=7)
    b = generate_dataset(n_series=3, length=30, seed=7)
    for sa, sb in zip(a, b):
        assert np.allclose(sa.values, sb.values)


# ----- metrics --------------------------------------------------------------
def test_metrics_zero_error():
    y = [1.0, 2.0, 3.0]
    assert mape(y, y) == pytest.approx(0.0, abs=1e-6)
    assert smape(y, y) == pytest.approx(0.0, abs=1e-6)
