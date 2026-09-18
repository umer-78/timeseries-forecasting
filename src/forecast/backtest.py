"""Rolling-origin evaluation.

A single train/test split scores a model on one arbitrary window, which is how a
model that got lucky once ends up in production. Rolling origin refits at many
cut-off points and averages, so the score reflects how the model behaves over
time rather than how one month happened to go.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .metrics import report
from .models import Model


@dataclass(frozen=True)
class Fold:
    """One cut-off: what the model saw, what it said, what actually happened."""

    origin: int
    actual: list[float]
    predicted: list[float]


@dataclass(frozen=True)
class Backtest:
    model: str
    horizon: int
    folds: list[Fold]
    scores: dict[str, float]

    @property
    def points(self) -> int:
        return sum(len(f.actual) for f in self.folds)


def rolling_origin(model: Model, series: Sequence[float], *, horizon: int = 6,
                   initial: int | None = None, step: int = 1,
                   expanding: bool = True, season_length: int = 1) -> Backtest:
    """Refit at each origin and score the next `horizon` points.

    expanding=True lets the training window grow, which is what a real pipeline
    does. expanding=False slides a fixed-width window instead, which is the right
    choice when old history has stopped being representative.
    """
    series = list(series)
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if step < 1:
        raise ValueError("step must be at least 1")

    initial = initial if initial is not None else max(2 * season_length, len(series) // 2)
    if initial + horizon > len(series):
        raise ValueError(
            f"a {initial}-point training window plus a {horizon}-point horizon "
            f"needs more than {len(series)} observations"
        )

    folds: list[Fold] = []
    origin = initial
    while origin + horizon <= len(series):
        train = series[:origin] if expanding else series[origin - initial:origin]
        actual = series[origin:origin + horizon]
        predicted = model.fit_predict(train, horizon)
        folds.append(Fold(origin, list(actual), list(predicted)))
        origin += step

    if not folds:
        raise ValueError("no folds — the series is too short for these settings")

    flat_actual = [v for f in folds for v in f.actual]
    flat_predicted = [v for f in folds for v in f.predicted]
    scores = report(flat_actual, flat_predicted, series[:initial], season_length)

    return Backtest(model.name, horizon, folds, scores)


def compare(models: Sequence[Model], series: Sequence[float], **kwargs) -> list[Backtest]:
    """Backtest several models on the same folds, best MASE first.

    Sorted by MASE where it is available, because it is the only measure here
    that says whether a model beat the seasonal naive baseline, which is the
    question that decides whether it is worth running at all.
    """
    results = [rolling_origin(model, series, **kwargs) for model in models]
    key = "mase" if all("mase" in r.scores for r in results) else "mae"
    return sorted(results, key=lambda r: r.scores[key])
