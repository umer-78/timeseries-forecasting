"""Accuracy measures for a forecast.

The one worth reading carefully is MASE. It is scaled by how well a seasonal
naive forecast did *on the training data*, not on the test window — scaling by
the test window makes the number depend on the thing being measured, and a model
can then look better simply because the test period was hard for everything.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Sequence


def _check(actual: Sequence[float], predicted: Sequence[float]) -> None:
    if len(actual) != len(predicted):
        raise ValueError(f"{len(actual)} actuals against {len(predicted)} forecasts")
    if not actual:
        raise ValueError("nothing to score")


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean absolute error, in the units of the series."""
    _check(actual, predicted)
    return sum(abs(a - p) for a, p in zip(actual, predicted, strict=True)) / len(actual)


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Root mean squared error. Punishes a few large misses more than MAE does."""
    _check(actual, predicted)
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted, strict=True)) / len(actual))


def mape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean absolute percentage error.

    Undefined wherever the actual is zero, so those points are skipped and the
    result is the mean over the rest. If every actual is zero there is no
    percentage to report and this raises rather than returning a made-up number.
    """
    _check(actual, predicted)
    pairs = [(a, p) for a, p in zip(actual, predicted, strict=True) if a != 0]
    if not pairs:
        raise ValueError("MAPE is undefined when every actual is zero — use sMAPE")
    return sum(abs((a - p) / a) for a, p in pairs) / len(pairs) * 100


def smape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Symmetric MAPE, bounded at 200% and defined at zero.

    Uses the (|a| + |p|) / 2 denominator. A point where both are zero counts as
    zero error rather than as a division by zero.
    """
    _check(actual, predicted)
    total = 0.0
    for a, p in zip(actual, predicted, strict=True):
        denominator = (abs(a) + abs(p)) / 2
        total += 0.0 if denominator == 0 else abs(a - p) / denominator
    return total / len(actual) * 100


def mase(actual: Sequence[float], predicted: Sequence[float], training: Sequence[float],
         season_length: int = 1) -> float:
    """Mean absolute scaled error.

    1.0 means "no better than a seasonal naive forecast fitted on the training
    data". Below 1 is an improvement on it; above 1 is worse. Being scale-free,
    it is the only one of these that can be averaged across different series.
    """
    _check(actual, predicted)
    if season_length < 1:
        raise ValueError("season length must be at least 1")
    if len(training) <= season_length:
        raise ValueError(f"need more than {season_length} training points to scale MASE")

    naive_errors = [abs(training[i] - training[i - season_length])
                    for i in range(season_length, len(training))]
    scale = sum(naive_errors) / len(naive_errors)
    if scale == 0:
        raise ValueError("the training series is perfectly seasonal; MASE would divide by zero")

    return mae(actual, predicted) / scale


def report(actual: Sequence[float], predicted: Sequence[float], training: Sequence[float],
           season_length: int = 1) -> dict[str, float]:
    """Every measure at once, with the ones that are undefined left out."""
    scores = {"mae": mae(actual, predicted), "rmse": rmse(actual, predicted),
              "smape": smape(actual, predicted)}
    for name, fn in (("mape", lambda: mape(actual, predicted)),
                     ("mase", lambda: mase(actual, predicted, training, season_length))):
        # A measure that is undefined for this data is left out rather than
        # reported as nan, which reads as a number and gets averaged as one.
        with contextlib.suppress(ValueError):
            scores[name] = fn()
    return scores
