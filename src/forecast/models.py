"""Forecasting models, from the baselines upward.

The baselines come first on purpose. A seasonal naive forecast — "next January
looks like last January" — is genuinely hard to beat on real business data, and a
model that cannot beat it is not worth deploying however sophisticated it is. The
whole point of MASE is to make that comparison the default reading.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


class Model:
    """A forecaster. fit() learns from history, predict(h) returns h steps."""

    name = "model"

    def fit(self, series: Sequence[float]) -> Model:
        raise NotImplementedError

    def predict(self, horizon: int) -> list[float]:
        raise NotImplementedError

    def fit_predict(self, series: Sequence[float], horizon: int) -> list[float]:
        return self.fit(series).predict(horizon)

    @staticmethod
    def _require(series: Sequence[float], minimum: int, name: str) -> None:
        if len(series) < minimum:
            raise ValueError(f"{name} needs at least {minimum} points, got {len(series)}")


@dataclass
class Naive(Model):
    """Tomorrow is today. The floor every other model has to clear."""

    name: str = "naive"
    _last: float = 0.0

    def fit(self, series: Sequence[float]) -> Naive:
        self._require(series, 1, "naive")
        self._last = series[-1]
        return self

    def predict(self, horizon: int) -> list[float]:
        return [self._last] * horizon


@dataclass
class SeasonalNaive(Model):
    """This January is last January. The baseline that actually wins sometimes."""

    season_length: int = 12
    name: str = "seasonal naive"
    _season: list[float] = field(default_factory=list)

    def fit(self, series: Sequence[float]) -> SeasonalNaive:
        self._require(series, self.season_length, "seasonal naive")
        self._season = list(series[-self.season_length:])
        return self

    def predict(self, horizon: int) -> list[float]:
        return [self._season[i % self.season_length] for i in range(horizon)]


@dataclass
class Drift(Model):
    """Draw a line through the first and last points and continue it."""

    name: str = "drift"
    _last: float = 0.0
    _slope: float = 0.0

    def fit(self, series: Sequence[float]) -> Drift:
        self._require(series, 2, "drift")
        self._last = series[-1]
        self._slope = (series[-1] - series[0]) / (len(series) - 1)
        return self

    def predict(self, horizon: int) -> list[float]:
        return [self._last + self._slope * (i + 1) for i in range(horizon)]


@dataclass
class Mean(Model):
    """The historical average, forever. Useful when a series has no trend at all."""

    name: str = "mean"
    _mean: float = 0.0

    def fit(self, series: Sequence[float]) -> Mean:
        self._require(series, 1, "mean")
        self._mean = sum(series) / len(series)
        return self

    def predict(self, horizon: int) -> list[float]:
        return [self._mean] * horizon


@dataclass
class SimpleExponentialSmoothing(Model):
    """Weighted average of the past, weights decaying by alpha. No trend, no season."""

    alpha: float = 0.3
    name: str = "ses"
    _level: float = 0.0

    def __post_init__(self) -> None:
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")

    def fit(self, series: Sequence[float]) -> SimpleExponentialSmoothing:
        self._require(series, 1, "ses")
        level = series[0]
        for value in series[1:]:
            level = self.alpha * value + (1 - self.alpha) * level
        self._level = level
        return self

    def predict(self, horizon: int) -> list[float]:
        return [self._level] * horizon


@dataclass
class Holt(Model):
    """Exponential smoothing with a trend. Flat forecasts become sloped ones."""

    alpha: float = 0.3
    beta: float = 0.1
    damped: float = 1.0        # < 1 flattens the trend as the horizon grows
    name: str = "holt"
    _level: float = 0.0
    _trend: float = 0.0

    def __post_init__(self) -> None:
        if not 0 < self.alpha <= 1 or not 0 < self.beta <= 1:
            raise ValueError("alpha and beta must be in (0, 1]")
        if not 0 < self.damped <= 1:
            raise ValueError("the damping factor must be in (0, 1]")

    def fit(self, series: Sequence[float]) -> Holt:
        self._require(series, 2, "holt")
        level, trend = series[0], series[1] - series[0]
        for value in series[1:]:
            previous = level
            level = self.alpha * value + (1 - self.alpha) * (level + self.damped * trend)
            trend = self.beta * (level - previous) + (1 - self.beta) * self.damped * trend
        self._level, self._trend = level, trend
        return self

    def predict(self, horizon: int) -> list[float]:
        out = []
        for h in range(1, horizon + 1):
            # Damped trend sums phi + phi^2 + ... + phi^h rather than h * trend.
            factor = sum(self.damped ** i for i in range(1, h + 1)) if self.damped < 1 else h
            out.append(self._level + factor * self._trend)
        return out


@dataclass
class HoltWinters(Model):
    """Level, trend and season together — additive seasonality.

    The initial season comes from averaging each position across whole periods,
    not from the first period alone: one January is a sample of size one.
    """

    season_length: int = 12
    alpha: float = 0.3
    beta: float = 0.1
    gamma: float = 0.2
    name: str = "holt-winters"
    _level: float = 0.0
    _trend: float = 0.0
    _season: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        for label, value in (("alpha", self.alpha), ("beta", self.beta), ("gamma", self.gamma)):
            if not 0 < value <= 1:
                raise ValueError(f"{label} must be in (0, 1]")
        if self.season_length < 2:
            raise ValueError("season length must be at least 2")

    def fit(self, series: Sequence[float]) -> HoltWinters:
        m = self.season_length
        self._require(series, 2 * m, "holt-winters")

        periods = len(series) // m
        averages = [sum(series[p * m:(p + 1) * m]) / m for p in range(periods)]
        season = [
            sum(series[p * m + i] - averages[p] for p in range(periods)) / periods
            for i in range(m)
        ]
        # Additive seasonal factors must sum to zero, or they compete with the level.
        offset = sum(season) / m
        season = [s - offset for s in season]

        level = averages[0]
        trend = (averages[-1] - averages[0]) / max(periods - 1, 1) / m

        for i, value in enumerate(series):
            index = i % m
            previous = level
            level = self.alpha * (value - season[index]) + (1 - self.alpha) * (level + trend)
            trend = self.beta * (level - previous) + (1 - self.beta) * trend
            season[index] = self.gamma * (value - level) + (1 - self.gamma) * season[index]

            if index == m - 1:
                # Renormalise at the end of each season. Additive factors that are
                # free to drift away from a zero mean compete with the level for
                # the same signal: the level absorbs the drift, the trend then
                # inherits the bias, and a long forecast runs low. Moving the mean
                # back into the level keeps the two separable.
                drift = sum(season) / m
                if drift:
                    season = [value - drift for value in season]
                    level += drift

        self._level, self._trend, self._season = level, trend, season
        self._offset = len(series) % m
        return self

    def predict(self, horizon: int) -> list[float]:
        m = self.season_length
        return [
            self._level + (h + 1) * self._trend + self._season[(self._offset + h) % m]
            for h in range(horizon)
        ]


def default_models(season_length: int = 12) -> list[Model]:
    """The set `forecast compare` runs, baselines first."""
    return [
        Naive(),
        SeasonalNaive(season_length=season_length),
        Mean(),
        Drift(),
        SimpleExponentialSmoothing(alpha=0.3),
        Holt(alpha=0.4, beta=0.1),
        Holt(alpha=0.4, beta=0.1, damped=0.9, name="holt (damped)"),
        HoltWinters(season_length=season_length, alpha=0.3, beta=0.05, gamma=0.3),
    ]
