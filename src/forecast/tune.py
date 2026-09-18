"""Choosing smoothing parameters instead of guessing them.

Fixed defaults like alpha=0.3 are a reasonable place to start and a poor place to
stop: on a series with a strong trend they leave the trend term chronically
behind, and the forecast runs low in a way no amount of extra history fixes. The
parameters are searched here on a rolling origin, so the score that picks them is
measured the same way the model will be used — never on data the fit has already
seen.

The search is scored at the horizon the caller actually forecasts, not at one
step. Parameters that win a one-step contest are not the parameters that win a
twelve-step one: a trend term that chases the last observation looks excellent
one step out and drifts badly by month twelve. On the sample data, tuning at one
step made the held-out year *worse* than the defaults; tuning at the real horizon
fixed most of that gap.

Tuning is not free improvement. On a short, noisy series a 216-point grid can fit
the search itself, which is why `auto` keeps the seasonal naive baseline whenever
the baseline is at least as good.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .backtest import rolling_origin
from .models import Holt, HoltWinters, Model, SeasonalNaive, SimpleExponentialSmoothing

# Coarse on purpose. A finer grid fits the search to the noise in one series and
# does not survive the next one.
GRID = (0.05, 0.15, 0.3, 0.5, 0.7, 0.9)


@dataclass(frozen=True)
class Tuned:
    model: Model
    parameters: dict[str, float]
    score: float
    evaluated: int


def _score(model: Model, series: Sequence[float], initial: int, season_length: int,
           horizon: int) -> float:
    try:
        return rolling_origin(model, series, horizon=horizon, initial=initial, step=1,
                              season_length=season_length).scores["mae"]
    except ValueError:
        return float("inf")


def search(factory: Callable[..., Model], series: Sequence[float], grids: dict[str, Sequence[float]],
           *, initial: int | None = None, season_length: int = 1, horizon: int = 1) -> Tuned:
    """Grid-search one model's parameters, scored at `horizon` steps ahead."""
    series = list(series)
    initial = initial if initial is not None else max(2 * season_length, len(series) // 2)

    names = list(grids)
    best: Tuned | None = None
    evaluated = 0

    def walk(index: int, chosen: dict[str, float]) -> None:
        nonlocal best, evaluated
        if index == len(names):
            model = factory(**chosen)
            score = _score(model, series, initial, season_length, horizon)
            evaluated += 1
            if best is None or score < best.score:
                best = Tuned(model, dict(chosen), score, evaluated)
            return
        for value in grids[names[index]]:
            walk(index + 1, {**chosen, names[index]: value})

    walk(0, {})
    if best is None or best.score == float("inf"):
        raise ValueError("no parameter combination could be scored on this series")

    # Refit the winner on everything, so the returned model is ready to predict.
    return Tuned(best.model.fit(series), best.parameters, best.score, evaluated)


def tune_holt(series: Sequence[float], **kwargs) -> Tuned:
    return search(lambda alpha, beta: Holt(alpha=alpha, beta=beta), series,
                  {"alpha": GRID, "beta": GRID}, **kwargs)


def tune_holt_winters(series: Sequence[float], season_length: int, **kwargs) -> Tuned:
    return search(
        lambda alpha, beta, gamma: HoltWinters(season_length=season_length, alpha=alpha,
                                               beta=beta, gamma=gamma),
        series, {"alpha": GRID, "beta": GRID, "gamma": GRID},
        season_length=season_length, **kwargs)


def tune_ses(series: Sequence[float], **kwargs) -> Tuned:
    return search(lambda alpha: SimpleExponentialSmoothing(alpha=alpha), series,
                  {"alpha": GRID}, **kwargs)


def auto(series: Sequence[float], season_length: int = 1, **kwargs) -> Tuned:
    """Pick a model and its parameters for a series.

    Seasonal data gets Holt-Winters; anything else gets Holt. Either is compared
    against the seasonal naive baseline, and the baseline wins any tie — a
    parameterless model that scores the same is the one to keep.
    """
    series = list(series)
    if season_length > 1 and len(series) >= 2 * season_length:
        candidate = tune_holt_winters(series, season_length, **kwargs)
    else:
        candidate = tune_holt(series, season_length=season_length, **kwargs)

    if season_length > 1 and len(series) > season_length:
        baseline = SeasonalNaive(season_length=season_length)
        initial = kwargs.get("initial") or max(2 * season_length, len(series) // 2)
        horizon = kwargs.get("horizon", 1)
        baseline_score = _score(baseline, series, initial, season_length, horizon)
        if baseline_score <= candidate.score:
            return Tuned(baseline.fit(series), {}, baseline_score, candidate.evaluated)

    return candidate
