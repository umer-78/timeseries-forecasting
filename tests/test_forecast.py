import math
from pathlib import Path

import pytest

from forecast import (
    Drift,
    Holt,
    HoltWinters,
    Mean,
    Naive,
    SeasonalNaive,
    SimpleExponentialSmoothing,
    compare,
    default_models,
    infer_season_length,
    mae,
    mape,
    mase,
    read_csv,
    report,
    rmse,
    rolling_origin,
    smape,
)
from forecast.data import parse_date
from forecast.tune import GRID, auto, tune_holt, tune_holt_winters, tune_ses

DATA = Path(__file__).resolve().parent.parent / "data"


def seasonal_trend(n: int, level: float = 100, slope: float = 2, amplitude: float = 20,
                   period: int = 12) -> list[float]:
    """A clean series with a trend and a season, and no noise."""
    return [level + slope * i + amplitude * math.sin(2 * math.pi * i / period) for i in range(n)]


# ------------------------------------------------------------------ metrics

def test_mae_and_rmse_on_hand_worked_numbers():
    actual, predicted = [10, 20, 30, 40], [12, 18, 33, 38]
    # errors are 2, 2, 3, 2 -> mean 2.25; squares 4, 4, 9, 4 -> mean 5.25
    assert mae(actual, predicted) == 2.25
    assert rmse(actual, predicted) == pytest.approx(math.sqrt(5.25))


def test_rmse_punishes_one_large_miss_more_than_mae_does():
    spread = [10, 10, 10, 10]
    even = [12, 12, 12, 12]
    lumpy = [10, 10, 10, 18]

    assert mae(spread, even) == mae(spread, lumpy) == 2.0
    assert rmse(spread, lumpy) > rmse(spread, even)


def test_a_perfect_forecast_scores_zero_everywhere():
    values = [3.0, 1.0, 4.0, 1.0, 5.0]
    assert mae(values, values) == 0
    assert rmse(values, values) == 0
    assert mape(values, values) == 0
    assert smape(values, values) == 0


def test_mape_skips_zeros_rather_than_dividing_by_them():
    # The zero actual is dropped; the remaining two are 10% and 10% out.
    assert mape([0, 100, 200], [5, 110, 180]) == pytest.approx(10.0)


def test_mape_refuses_a_series_that_is_all_zeros():
    with pytest.raises(ValueError, match="undefined when every actual is zero"):
        mape([0, 0], [1, 2])


def test_smape_is_defined_at_zero_and_bounded_at_two_hundred():
    assert smape([0, 0], [0, 0]) == 0
    assert smape([0], [10]) == pytest.approx(200.0)
    assert smape([100], [0]) == pytest.approx(200.0)


def test_mase_is_scaled_by_the_training_window_not_the_test_window():
    training = [10, 12, 14, 16, 18, 20]        # seasonal naive error of 2 per step
    actual, predicted = [22, 24], [23, 25]     # absolute error of 1 per step

    assert mase(actual, predicted, training, season_length=1) == pytest.approx(0.5)

    # A harder test window does not change the scale; only the numerator moves.
    assert mase([22, 24], [26, 28], training, 1) == pytest.approx(2.0)


def test_mase_of_one_means_no_better_than_the_baseline():
    training = [10, 12, 14, 16, 18, 20]
    assert mase([22, 24], [20, 22], training, 1) == pytest.approx(1.0)


def test_mase_refuses_a_training_window_too_short_to_scale():
    with pytest.raises(ValueError, match="need more than"):
        mase([1, 2], [1, 2], [5], season_length=1)


def test_mase_refuses_a_perfectly_seasonal_training_window():
    with pytest.raises(ValueError, match="divide by zero"):
        mase([1], [1], [5, 5, 5, 5], season_length=1)


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="actuals against"):
        mae([1, 2, 3], [1, 2])


def test_report_leaves_out_the_measures_it_cannot_compute():
    scores = report([0, 0, 0], [1, 1, 1], [5, 5, 5, 5], season_length=1)

    assert "mae" in scores and "smape" in scores
    assert "mape" not in scores   # every actual is zero
    assert "mase" not in scores   # the training window has no variation


# ------------------------------------------------------------------ models

def test_naive_repeats_the_last_value():
    assert Naive().fit_predict([5, 9, 7], 3) == [7, 7, 7]


def test_seasonal_naive_repeats_the_last_whole_season():
    season = [1, 2, 3, 4]
    assert SeasonalNaive(season_length=4).fit_predict([9, 9] + season, 6) == [1, 2, 3, 4, 1, 2]


def test_mean_forecasts_the_average():
    assert Mean().fit_predict([2, 4, 6], 2) == [4, 4]


def test_drift_continues_the_line_between_the_first_and_last_points():
    # 10 to 20 over 5 steps is a slope of 2.
    assert Drift().fit_predict([10, 11, 13, 17, 20, 20], 3) == [22, 24, 26]


def test_exponential_smoothing_tracks_a_level_shift_without_jumping_to_it():
    flat_then_up = [10] * 10 + [20] * 3
    forecast = SimpleExponentialSmoothing(alpha=0.3).fit_predict(flat_then_up, 1)[0]

    assert 10 < forecast < 20, "SES should move toward the new level, not leap to it"


def test_a_larger_alpha_reacts_faster():
    series = [10] * 10 + [20] * 3
    slow = SimpleExponentialSmoothing(alpha=0.1).fit_predict(series, 1)[0]
    fast = SimpleExponentialSmoothing(alpha=0.8).fit_predict(series, 1)[0]

    assert fast > slow


def test_holt_extends_a_straight_line_almost_exactly():
    line = [float(10 + 3 * i) for i in range(30)]
    forecast = Holt(alpha=0.6, beta=0.3).fit_predict(line, 3)

    assert forecast == pytest.approx([100, 103, 106], rel=0.02)


def test_damping_flattens_a_long_horizon():
    line = [float(10 + 3 * i) for i in range(30)]
    straight = Holt(alpha=0.6, beta=0.3).fit_predict(line, 12)[-1]
    damped = Holt(alpha=0.6, beta=0.3, damped=0.8).fit_predict(line, 12)[-1]

    assert damped < straight


def test_fixed_smoothing_parameters_leave_the_trend_behind():
    """The reason tuning exists, pinned as a fact rather than left as folklore.

    On a series with a constant slope of 2, a hand-picked beta recovers a trend
    well under it, and the forecast runs low by more with every step.
    """
    model = HoltWinters(season_length=12, alpha=0.4, beta=0.1, gamma=0.3).fit(seasonal_trend(60))

    assert model._trend < 1.6, "expected the trend to lag the true slope of 2"


def test_tuning_recovers_a_clean_trend_and_season_almost_exactly():
    series = seasonal_trend(60)
    tuned = tune_holt_winters(series, season_length=12)
    expected = seasonal_trend(72)[60:]

    assert mae(expected, tuned.model.predict(12)) < 0.1
    assert tuned.evaluated == len(GRID) ** 3


def test_tuning_beats_the_hand_picked_defaults_on_the_same_series():
    series = seasonal_trend(60)
    expected = seasonal_trend(72)[60:]

    fixed = HoltWinters(season_length=12, alpha=0.4, beta=0.1, gamma=0.3).fit_predict(series, 12)
    tuned = tune_holt_winters(series, season_length=12).model.predict(12)

    assert mae(expected, tuned) < mae(expected, fixed) / 10


def test_holt_winters_seasonal_factors_stay_centred_on_zero():
    """Factors free to drift compete with the level for the same signal, so they
    are renormalised at the end of every season and the mean is handed back to
    the level."""
    model = HoltWinters(season_length=12).fit(seasonal_trend(60))
    assert sum(model._season) == pytest.approx(0, abs=1e-9)


def test_renormalising_the_season_does_not_change_the_forecast():
    """It is an identifiability fix, not an accuracy one: level + season is
    unchanged, so predictions must be identical to the unnormalised recursion."""
    series = seasonal_trend(60, level=500)
    model = HoltWinters(season_length=12, alpha=0.4, beta=0.1, gamma=0.3).fit(series)

    rebuilt = [model._level + (h + 1) * model._trend + model._season[(model._offset + h) % 12]
               for h in range(12)]
    assert model.predict(12) == pytest.approx(rebuilt)


def test_models_refuse_a_series_too_short_to_fit():
    with pytest.raises(ValueError, match="at least 12 points"):
        SeasonalNaive(season_length=12).fit([1, 2, 3])
    with pytest.raises(ValueError, match="at least 24 points"):
        HoltWinters(season_length=12).fit(seasonal_trend(20))
    with pytest.raises(ValueError, match="at least 2 points"):
        Drift().fit([1])


def test_smoothing_parameters_are_validated():
    for bad in (0, -0.1, 1.5):
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            SimpleExponentialSmoothing(alpha=bad)
    with pytest.raises(ValueError, match="season length"):
        HoltWinters(season_length=1)


# ------------------------------------------------------------------ backtesting

def test_rolling_origin_refits_at_every_origin():
    series = seasonal_trend(48)
    result = rolling_origin(Naive(), series, horizon=6, initial=24, step=6, season_length=12)

    assert [f.origin for f in result.folds] == [24, 30, 36, 42]
    assert result.points == 24
    assert all(len(f.predicted) == 6 for f in result.folds)


def test_a_fold_never_sees_the_future_it_is_scored_on():
    series = list(range(40))
    seen = []

    class Spy(Naive):
        def fit(self, values):
            seen.append(list(values))
            return super().fit(values)

    rolling_origin(Spy(), series, horizon=4, initial=20, step=4, season_length=1)

    for window, origin in zip(seen, [20, 24, 28, 32, 36], strict=False):
        assert window == series[:origin], "training data leaked past the origin"


def test_an_expanding_window_grows_and_a_sliding_one_does_not():
    series = seasonal_trend(60)
    widths = []

    class Spy(Naive):
        def fit(self, values):
            widths.append(len(values))
            return super().fit(values)

    rolling_origin(Spy(), series, horizon=6, initial=24, step=6, season_length=12)
    assert widths == sorted(widths) and widths[0] < widths[-1]

    widths.clear()
    rolling_origin(Spy(), series, horizon=6, initial=24, step=6, expanding=False, season_length=12)
    assert len(set(widths)) == 1


def test_a_series_too_short_for_the_settings_says_so():
    with pytest.raises(ValueError, match="needs more than"):
        rolling_origin(Naive(), seasonal_trend(20), horizon=6, initial=18)


def test_compare_puts_the_best_model_first_and_the_mean_last():
    results = compare(default_models(12), seasonal_trend(72), horizon=6, initial=36,
                      step=3, season_length=12)

    assert results[0].model == "holt-winters"
    assert results[-1].model == "mean"
    assert results[0].scores["mase"] < results[-1].scores["mase"]


def test_every_model_is_scored_on_identical_folds():
    results = compare(default_models(12), seasonal_trend(72), horizon=6, initial=36,
                      step=3, season_length=12)

    origins = {tuple(f.origin for f in r.folds) for r in results}
    assert len(origins) == 1, "models were compared on different windows"


# ------------------------------------------------------------------ data

def test_reading_the_sample_monthly_series():
    series = read_csv(DATA / "monthly-sales.csv")

    assert len(series) == 72
    assert series.name == "revenue"
    assert series.dates[0].isoformat() == "2020-01-01"
    assert series.span == "2020-01-01 to 2025-12-01"


def test_monthly_data_infers_a_period_of_twelve():
    assert infer_season_length(read_csv(DATA / "monthly-sales.csv")) == 12


def test_daily_data_infers_a_weekly_period():
    assert infer_season_length(read_csv(DATA / "daily-traffic.csv")) == 7


def test_rows_out_of_order_are_sorted_by_date(tmp_path):
    path = tmp_path / "s.csv"
    path.write_text("date,value\n2024-03-01,3\n2024-01-01,1\n2024-02-01,2\n", encoding="utf-8")

    series = read_csv(path)

    assert series.values == [1, 2, 3]
    assert series.dates == sorted(series.dates)


def test_several_date_formats_are_accepted():
    assert parse_date("2024-01-05").isoformat() == "2024-01-05"
    assert parse_date("05/01/2024").isoformat() == "2024-01-05"   # day first
    assert parse_date("2024-01").isoformat() == "2024-01-01"


def test_an_unreadable_date_names_the_value_and_the_formats():
    with pytest.raises(ValueError, match="unrecognised date"):
        parse_date("last Tuesday")


def test_a_non_numeric_value_names_the_line(tmp_path):
    path = tmp_path / "s.csv"
    path.write_text("date,value\n2024-01-01,1\n2024-02-01,oops\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 3"):
        read_csv(path)


def test_a_file_with_one_observation_is_not_a_series(tmp_path):
    path = tmp_path / "s.csv"
    path.write_text("date,value\n2024-01-01,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="at least 2"):
        read_csv(path)


# ------------------------------------------------------------------ end to end

def test_holt_winters_beats_every_baseline_on_the_sample_data():
    series = read_csv(DATA / "monthly-sales.csv")
    results = compare(default_models(12), series.values, horizon=6, initial=36,
                      step=3, season_length=12)

    assert results[0].model == "holt-winters"
    assert results[0].scores["mase"] < 0.5, "should be well clear of the naive baseline"
    assert results[0].scores["smape"] < 5


# ------------------------------------------------------------------ tuning

def test_tuning_searches_the_whole_grid():
    tuned = tune_ses(seasonal_trend(40, amplitude=0), season_length=1)

    assert tuned.evaluated == len(GRID)
    assert tuned.parameters["alpha"] in GRID


def test_holt_tuning_searches_two_parameters():
    tuned = tune_holt([float(10 + 3 * i) for i in range(40)], season_length=1)

    assert tuned.evaluated == len(GRID) ** 2
    assert set(tuned.parameters) == {"alpha", "beta"}


def test_the_tuned_model_comes_back_fitted_on_everything():
    series = seasonal_trend(48)
    tuned = tune_holt_winters(series, season_length=12)

    # Predicting immediately must work: a caller should not have to refit.
    assert len(tuned.model.predict(6)) == 6


def test_auto_picks_holt_winters_for_seasonal_data():
    assert auto(seasonal_trend(60), season_length=12).model.name == "holt-winters"


def test_auto_falls_back_to_holt_without_a_season():
    straight = [float(10 + 3 * i) for i in range(40)]
    assert auto(straight, season_length=1).model.name == "holt"


def test_auto_keeps_the_baseline_when_nothing_beats_it():
    """Pure repetition with no trend: seasonal naive is exactly right, and a
    smoothing model can only match it. A parameterless model that ties is the one
    to keep, so the baseline wins ties rather than losing them to something that
    looks more impressive."""
    repeated = [10, 20, 30, 40] * 12
    assert auto(repeated, season_length=4, horizon=4).model.name == "seasonal naive"


def test_tuning_at_the_real_horizon_beats_tuning_at_one_step():
    """The reason `search` takes a horizon at all.

    Parameters that win a one-step contest chase the last observation: excellent
    next month, adrift by month twelve. Scored on a held-out year of the sample
    data, tuning at one step is materially worse than tuning at twelve — which is
    the mistake this parameter exists to prevent.
    """
    series = read_csv(DATA / "monthly-sales.csv").values
    train, test = series[:60], series[60:]

    one_step = tune_holt_winters(train, season_length=12, horizon=1).model.predict(12)
    full = tune_holt_winters(train, season_length=12, horizon=12).model.predict(12)

    assert mae(test, full) < mae(test, one_step)


def test_tuning_does_not_always_beat_sensible_defaults():
    """Stated as a test rather than left as a surprise.

    Sixty points of noisy monthly data is not much to search 216 combinations
    against, and here the hand-picked defaults generalise better than any of them.
    That is why `auto` keeps the baseline on ties and why `compare` still prints
    every model rather than announcing a winner.
    """
    series = read_csv(DATA / "monthly-sales.csv").values
    train, test = series[:60], series[60:]

    fixed = HoltWinters(season_length=12).fit_predict(train, 12)
    tuned = tune_holt_winters(train, season_length=12, horizon=12).model.predict(12)

    assert mae(test, fixed) < mae(test, tuned)
