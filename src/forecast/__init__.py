"""Time series forecasting: baselines, exponential smoothing, and honest backtesting."""

from .backtest import Backtest, Fold, compare, rolling_origin
from .data import Series, infer_season_length, read_csv
from .metrics import mae, mape, mase, report, rmse, smape
from .models import (
    Drift,
    Holt,
    HoltWinters,
    Mean,
    Model,
    Naive,
    SeasonalNaive,
    SimpleExponentialSmoothing,
    default_models,
)

__all__ = [
    "Backtest", "Drift", "Fold", "Holt", "HoltWinters", "Mean", "Model", "Naive",
    "SeasonalNaive", "Series", "SimpleExponentialSmoothing", "compare", "default_models",
    "infer_season_length", "mae", "mape", "mase", "read_csv", "report", "rmse",
    "rolling_origin", "smape",
]
__version__ = "1.0.0"
