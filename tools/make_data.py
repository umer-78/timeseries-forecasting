"""Writes the sample series in data/.

Seeded, so re-running reproduces the files byte for byte and the figures in the
README stay true. CI regenerates them and fails if anything differs.
"""

from __future__ import annotations

import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 20260918
DATA = Path(__file__).resolve().parent.parent / "data"


def monthly_sales() -> list[tuple[str, float]]:
    """Six years of monthly retail revenue: growth, a December peak, a February dip."""
    rng = random.Random(SEED)
    rows = []
    month = date(2020, 1, 1)
    for i in range(72):
        trend = 42_000 + 520 * i
        season = {
            0: -0.06, 1: -0.11, 2: 0.01, 3: 0.03, 4: 0.05, 5: 0.02,
            6: -0.02, 7: -0.03, 8: 0.04, 9: 0.06, 10: 0.14, 11: 0.31,
        }[month.month - 1]
        value = trend * (1 + season) * (1 + rng.gauss(0, 0.025))
        rows.append((month.isoformat(), round(value, 2)))
        month = date(month.year + (month.month == 12), month.month % 12 + 1, 1)
    return rows


def daily_traffic() -> list[tuple[str, float]]:
    """Two years of daily sessions: a weekly rhythm and a slow climb."""
    rng = random.Random(SEED + 1)
    rows = []
    day = date(2024, 1, 1)
    for i in range(730):
        trend = 8_400 + 4.2 * i
        weekday = [1.08, 1.10, 1.09, 1.06, 0.98, 0.74, 0.70][day.weekday()]
        # A gentle yearly wave on top of the weekly one.
        yearly = 1 + 0.07 * math.sin(2 * math.pi * i / 365.25)
        value = trend * weekday * yearly * (1 + rng.gauss(0, 0.04))
        rows.append((day.isoformat(), round(value)))
        day += timedelta(days=1)
    return rows


def write(name: str, rows: list[tuple[str, float]], value_header: str) -> None:
    path = DATA / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["date", value_header])
        writer.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def main() -> None:
    DATA.mkdir(exist_ok=True)
    write("monthly-sales.csv", monthly_sales(), "revenue")
    write("daily-traffic.csv", daily_traffic(), "sessions")


if __name__ == "__main__":
    main()
