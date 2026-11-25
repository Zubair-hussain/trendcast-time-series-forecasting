"""Minimal end-to-end example."""
from trendcast import TrendCast, generate_dataset

# 1. get data (or load your own 1-D list of numbers)
series = generate_dataset(n_series=1, length=120)[0].values

# 2. fit + forecast the next 14 steps
fc = TrendCast().fit(series[:-14]).forecast(horizon=14)

# 3. use it
print("label    :", fc.label)          # rising / steady / fading
print("momentum :", round(fc.momentum, 1))
print("next 14  :", [round(float(v), 1) for v in fc.point])
