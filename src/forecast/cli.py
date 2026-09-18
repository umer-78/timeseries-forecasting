"""forecast: fit, predict and backtest a series from a CSV."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .backtest import compare, rolling_origin
from .data import infer_season_length, read_csv
from .models import Model, default_models
from .tune import auto


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forecast", description=__doc__)
    parser.add_argument("--version", action="version", version=f"forecast {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("csv", type=Path, help="a CSV with a date column and a value column")
        p.add_argument("--date-column", default=None)
        p.add_argument("--value-column", default=None)
        p.add_argument("--season", type=int, default=None,
                       help="seasonal period; inferred from the dates when omitted")

    describe = sub.add_parser("describe", help="what is in the file")
    common(describe)

    predict = sub.add_parser("predict", help="forecast the next h points")
    common(predict)
    predict.add_argument("-h2", "--horizon", type=int, default=6)
    predict.add_argument("--json", action="store_true")

    back = sub.add_parser("backtest", help="rolling-origin score for one model")
    common(back)
    back.add_argument("-h2", "--horizon", type=int, default=6)
    back.add_argument("--initial", type=int, default=None)
    back.add_argument("--step", type=int, default=1)
    back.add_argument("--sliding", action="store_true", help="fixed-width window instead of expanding")

    comp = sub.add_parser("compare", help="backtest every model against the baselines")
    common(comp)
    comp.add_argument("-h2", "--horizon", type=int, default=6)
    comp.add_argument("--initial", type=int, default=None)
    comp.add_argument("--step", type=int, default=1)
    comp.add_argument("--json", action="store_true")

    tune = sub.add_parser("tune", help="grid-search smoothing parameters for this series")
    common(tune)
    tune.add_argument("-h2", "--horizon", type=int, default=6,
                      help="the horizon to score the search at — use the one you will forecast")
    tune.add_argument("--initial", type=int, default=None)
    tune.add_argument("--json", action="store_true")

    chart = sub.add_parser("chart", help="write a forecast chart and a backtest chart")
    common(chart)
    chart.add_argument("-h2", "--horizon", type=int, default=12)
    chart.add_argument("--initial", type=int, default=None)
    chart.add_argument("-o", "--out", type=Path, default=Path("reports"))

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Wraps the real work so that piping into `head` — which closes
    the pipe early — ends quietly instead of printing a BrokenPipeError."""
    try:
        return _run(argv)
    except BrokenPipeError:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        return 0
    except KeyboardInterrupt:
        print(file=sys.stderr)
        return 130
    except (ValueError, FileNotFoundError) as error:
        print(f"forecast: {error}", file=sys.stderr)
        return 2


def _run(argv: list[str] | None) -> int:
    args = build_parser().parse_args(argv)
    series = read_csv(args.csv, date_column=args.date_column, value_column=args.value_column)
    season = args.season if args.season else infer_season_length(series)

    if args.cmd == "describe":
        values = series.values
        print(f"{series.name}: {len(series)} observations, {series.span}")
        print(f"  seasonal period {season}" + ("  (inferred)" if not args.season else ""))
        print(f"  min {min(values):,.2f}   mean {sum(values) / len(values):,.2f}   max {max(values):,.2f}")
        print(f"  first {values[0]:,.2f}   last {values[-1]:,.2f}   "
              f"change {(values[-1] / values[0] - 1) * 100:+.1f}%")
        return 0

    if args.cmd == "tune":
        tuned = auto(series.values, season_length=season, horizon=args.horizon,
                     initial=args.initial)
        if args.json:
            print(json.dumps({"model": tuned.model.name, "parameters": tuned.parameters,
                              "scored_at_horizon": args.horizon,
                              "rolling_mae": round(tuned.score, 4),
                              "combinations_tried": tuned.evaluated}, indent=2))
            return 0
        print(f"{tuned.model.name} — {tuned.evaluated} combinations scored "
              f"at {args.horizon} step(s) ahead")
        for name, value in tuned.parameters.items():
            print(f"  {name:<6} {value}")
        if not tuned.parameters:
            print("  (no parameters — the baseline was not beaten)")
        print(f"  rolling MAE {tuned.score:,.2f}")
        return 0

    if args.cmd == "predict":
        tuned = auto(series.values, season_length=season, horizon=args.horizon)
        model: Model = tuned.model
        predictions = model.predict(args.horizon)
        if args.json:
            print(json.dumps({"model": model.name, "parameters": tuned.parameters,
                              "horizon": args.horizon,
                              "forecast": [round(v, 2) for v in predictions]}, indent=2))
            return 0
        chosen = ", ".join(f"{k}={v}" for k, v in tuned.parameters.items()) or "no parameters"
        print(f"{model.name} ({chosen}), {args.horizon} step(s) ahead")
        for i, value in enumerate(predictions, start=1):
            print(f"  +{i:<3} {value:>14,.2f}")
        return 0

    if args.cmd == "chart":
        from .plots import backtest_chart, forecast_chart

        tuned = auto(series.values, season_length=season, horizon=args.horizon,
                     initial=args.initial)
        first = forecast_chart(series, tuned.model.predict(args.horizon),
                               args.out / "forecast.png")
        result = rolling_origin(tuned.model, series.values, horizon=args.horizon,
                                initial=args.initial, step=max(args.horizon // 2, 1),
                                season_length=season)
        second = backtest_chart(series, result, args.out / "backtest.png")
        print(f"wrote {first}")
        print(f"wrote {second}")
        return 0

    kwargs = dict(horizon=args.horizon, initial=args.initial, step=args.step,
                  season_length=season)

    if args.cmd == "backtest":
        model = auto(series.values, season_length=season, horizon=args.horizon).model
        result = rolling_origin(model, series.values, expanding=not args.sliding, **kwargs)
        window = "sliding" if args.sliding else "expanding"
        print(f"{result.model}: {len(result.folds)} folds x {result.horizon} steps "
              f"= {result.points} scored points ({window} window)")
        for name, value in result.scores.items():
            print(f"  {name.upper():<6} {value:>10.3f}")
        return 0

    results = compare(default_models(season), series.values, **kwargs)
    if args.json:
        print(json.dumps([{"model": r.model, **{k: round(v, 4) for k, v in r.scores.items()}}
                          for r in results], indent=2))
        return 0

    folds = len(results[0].folds)
    print(f"{folds} folds x {results[0].horizon} steps, seasonal period {season}")
    print(f"{'model':<18}{'MASE':>8}{'MAE':>12}{'RMSE':>12}{'sMAPE':>9}")
    for result in results:
        s = result.scores
        # The seasonal naive row is the yardstick itself, so flagging it as
        # beating the yardstick would be nonsense. Its MASE is not exactly 1.0
        # because the scale comes from the training window while the score comes
        # from the test folds — different periods, on purpose.
        beats = s.get("mase", 9) < 1 and result.model != "seasonal naive"
        flag = "  <-- better than the naive baseline" if beats else ""
        print(f"{result.model:<18}{s.get('mase', float('nan')):>8.3f}{s['mae']:>12,.2f}"
              f"{s['rmse']:>12,.2f}{s['smape']:>8.1f}%{flag}")
    return 0
