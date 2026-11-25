"""
TrendCast — an original, lightweight trend-forecasting engine.

Author: ZH (replace with your name)
License: MIT
Portfolio: https://your-portfolio.example   (replace with your real URL)
"""

from .model import TrendCast, TrendForecast
from .dataset import load_dataset, generate_dataset

__all__ = ["TrendCast", "TrendForecast", "load_dataset", "generate_dataset"]
__version__ = "1.0.0"
