"""Charts. matplotlib is optional — everything else works without it."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

from .backtest import Backtest
from .data import Series

STYLE = {
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
}


def _pyplot():
    try:
        import matplotlib
    except ImportError as error:  # pragma: no cover - exercised only without matplotlib
        raise ImportError("charts need matplotlib: pip install 'forecast[charts]'") from error
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _future_dates(series: Series, horizon: int) -> list[date]:
    """Continue the series' own spacing rather than assuming calendar months."""
    step = (series.dates[-1] - series.dates[-2]).days
    if 28 <= step <= 31:
        out, current = [], series.dates[-1]
        for _ in range(horizon):
            current = date(current.year + (current.month == 12), current.month % 12 + 1, 1)
            out.append(current)
        return out
    return [series.dates[-1] + timedelta(days=step * (i + 1)) for i in range(horizon)]


def forecast_chart(series: Series, predicted: Sequence[float], path: str | Path,
                   title: str = "") -> Path:
    """History and forecast on one axis, with the join marked."""
    plt = _pyplot()
    future = _future_dates(series, len(predicted))

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.plot(series.dates, series.values, color="#2563eb", lw=1.4, label="history")
        # Start the forecast line at the last observation so the two connect.
        ax.plot([series.dates[-1], *future], [series.values[-1], *predicted],
                color="#f97316", lw=1.8, ls="--", label="forecast")
        ax.axvline(series.dates[-1], color="#9aa3b2", lw=0.8, ls=":")
        ax.set_title(title or f"{series.name}: {len(predicted)} steps ahead")
        ax.set_ylabel(series.name)
        ax.legend(frameon=False, loc="upper left")
        fig.autofmt_xdate()
        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=140)
        plt.close(fig)
    return Path(path)


def backtest_chart(series: Series, result: Backtest, path: str | Path) -> Path:
    """Every fold's forecast drawn over the actuals, so the misses are visible."""
    plt = _pyplot()

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.plot(series.dates, series.values, color="#16181d", lw=1.2, label="actual")
        for i, fold in enumerate(result.folds):
            span = series.dates[fold.origin:fold.origin + len(fold.predicted)]
            ax.plot(span, fold.predicted, color="#f97316", lw=1.1, alpha=0.75,
                    label="forecast per fold" if i == 0 else None)
        ax.set_title(f"{result.model}: {len(result.folds)} rolling origins, "
                     f"{result.horizon} steps each")
        ax.set_ylabel(series.name)
        ax.legend(frameon=False, loc="upper left")
        fig.autofmt_xdate()
        fig.tight_layout()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=140)
        plt.close(fig)
    return Path(path)
