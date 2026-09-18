"""Reading a series out of a CSV."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

FORMATS = ("%Y-%m-%d", "%Y-%m", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y")


@dataclass(frozen=True)
class Series:
    """Dates and values, oldest first."""

    dates: list[date]
    values: list[float]
    name: str = "series"

    def __len__(self) -> int:
        return len(self.values)

    def __post_init__(self) -> None:
        if len(self.dates) != len(self.values):
            raise ValueError("dates and values are different lengths")

    @property
    def span(self) -> str:
        if not self.dates:
            return "empty"
        return f"{self.dates[0].isoformat()} to {self.dates[-1].isoformat()}"


def parse_date(text: str) -> date:
    """Try the formats in order. Guessing per row is how a day becomes a month."""
    for fmt in FORMATS:
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognised date {text!r} — expected one of {', '.join(FORMATS)}")


def read_csv(path: str | Path, *, date_column: str | None = None,
             value_column: str | None = None) -> Series:
    """Read a two-column CSV. The columns are taken by name, or by position."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        raise ValueError(f"{path.name} is empty")

    header = [h.strip() for h in rows[0]]
    date_index = header.index(date_column) if date_column else 0
    value_index = header.index(value_column) if value_column else 1

    dates: list[date] = []
    values: list[float] = []
    for line, row in enumerate(rows[1:], start=2):
        if not row or not row[date_index].strip():
            continue
        try:
            values.append(float(row[value_index]))
        except ValueError:
            raise ValueError(f"{path.name} line {line}: {row[value_index]!r} is not a number") from None
        dates.append(parse_date(row[date_index]))

    if len(values) < 2:
        raise ValueError(f"{path.name} holds {len(values)} observation(s); a series needs at least 2")

    if dates != sorted(dates):
        order = sorted(range(len(dates)), key=lambda i: dates[i])
        dates = [dates[i] for i in order]
        values = [values[i] for i in order]

    return Series(dates, values, name=header[value_index])


def infer_season_length(series: Series) -> int:
    """Guess the seasonal period from the spacing of the dates.

    Monthly data is 12, quarterly 4, weekly 52, daily 7. Anything else returns 1,
    which means "no seasonality assumed" rather than a number picked hopefully.
    """
    if len(series) < 3:
        return 1
    gaps = [(series.dates[i + 1] - series.dates[i]).days for i in range(len(series) - 1)]
    typical = sorted(gaps)[len(gaps) // 2]
    if 28 <= typical <= 31:
        return 12
    if 89 <= typical <= 92:
        return 4
    if typical == 7:
        return 52
    if typical == 1:
        return 7
    return 1
