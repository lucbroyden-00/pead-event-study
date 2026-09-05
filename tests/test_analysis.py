import numpy as np
import pandas as pd
import pytest

from pead.analysis import (
    build_car_panel,
    caar_by_decile,
    cost_sensitivity,
    long_short_test,
)


def _single_event_panel_inputs(seed: int = 0):
    """One event's worth of abnormal returns over the full (0, +60) window,
    plus its matching decile-assigned events row."""
    rng = np.random.default_rng(seed)
    event_times = list(range(-1, 61))
    abnormal_returns = pd.DataFrame(
        {
            "cik": 1,
            "ticker": "AAA",
            "event_id": 0,
            "event_time": event_times,
            "abnormal_return": rng.normal(scale=0.01, size=len(event_times)),
        }
    )
    events = pd.DataFrame(
        {
            "event_id": [0],
            "decile": [7],
            "announcement_date": [pd.Timestamp("2024-03-01")],
        }
    )
    return events, abnormal_returns


def test_car_at_horizon_zero_equals_day_zero_abnormal_return():
    events, abnormal_returns = _single_event_panel_inputs()

    panel = build_car_panel(events, abnormal_returns)

    day0_ar = abnormal_returns.loc[abnormal_returns["event_time"] == 0, "abnormal_return"].iloc[0]
    car_at_0 = panel.loc[panel["event_time"] == 0, "car"].iloc[0]
    assert car_at_0 == pytest.approx(day0_ar)


def test_window_cars_match_manual_sums_of_daily_abnormal_returns():
    events, abnormal_returns = _single_event_panel_inputs()
    ar = abnormal_returns.set_index("event_time")["abnormal_return"]

    panel = build_car_panel(events, abnormal_returns)

    expected_announcement = ar.loc[0:1].sum()
    expected_drift = ar.loc[2:60].sum()
    assert panel["car_announcement"].iloc[0] == pytest.approx(expected_announcement)
    assert panel["car_drift"].iloc[0] == pytest.approx(expected_drift)


def test_caar_by_decile_averages_car_within_decile_at_each_event_time():
    panel = pd.DataFrame(
        {
            "event_id": [0, 0, 1, 1],
            "decile": [1, 1, 1, 1],
            "announcement_date": pd.Timestamp("2024-01-01"),
            "event_time": [0, 1, 0, 1],
            "car": [0.01, 0.02, 0.03, 0.06],
            "car_announcement": [0.02, 0.02, 0.06, 0.06],
            "car_drift": [0.0, 0.0, 0.0, 0.0],
        }
    )

    caar = caar_by_decile(panel)

    assert caar.loc[0, 1] == pytest.approx((0.01 + 0.03) / 2)
    assert caar.loc[1, 1] == pytest.approx((0.02 + 0.06) / 2)


def _long_short_panel(n_dates: int, events_per_decile_per_date: int, date_shock_scale: float, seed: int):
    """Synthetic event-level panel (decile 1 and 10 only) where every event
    sharing an announcement_date and decile shares that date/decile's shock,
    plus tiny idiosyncratic noise -- i.e. deliberately correlated within
    announcement_date. The decile-1 and decile-10 shocks on a given date are
    drawn independently (not a single shared shock), so they don't cancel out
    of the decile-10-minus-decile-1 spread -- a shared shock would fully net
    out of that contrast and *understate* the true correlation's effect on
    it, which is not what this is meant to demonstrate."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="7D")

    rows = []
    event_id = 0
    for date in dates:
        date_shocks = {1: rng.normal(scale=date_shock_scale), 10: rng.normal(scale=date_shock_scale)}
        for decile in (1, 10):
            decile_effect = -0.02 if decile == 1 else 0.02
            for _ in range(events_per_decile_per_date):
                idiosyncratic = rng.normal(scale=0.0005)
                rows.append(
                    {
                        "event_id": event_id,
                        "decile": decile,
                        "announcement_date": date,
                        "car_drift": decile_effect + date_shocks[decile] + idiosyncratic,
                    }
                )
                event_id += 1
    return pd.DataFrame(rows)


def test_clustered_standard_error_exceeds_naive_on_correlated_event_dates():
    panel = _long_short_panel(
        n_dates=10, events_per_decile_per_date=20, date_shock_scale=0.05, seed=0
    )

    result = long_short_test(panel)

    # spread = 2 * regression coefficient (the +1/-1 indicator coding halves
    # it), so se = |coefficient| / |t| = (spread / 2) / |t| for either fit.
    half_spread = result.loc[0, "spread"] / 2
    naive_se = abs(half_spread) / abs(result.loc[0, "naive_t_stat"])
    clustered_se = abs(half_spread) / abs(result.loc[0, "clustered_t_stat"])

    assert clustered_se > naive_se


def test_long_short_test_spread_matches_decile_group_means():
    panel = _long_short_panel(
        n_dates=5, events_per_decile_per_date=10, date_shock_scale=0.01, seed=1
    )

    result = long_short_test(panel)

    manual_mean_d10 = panel.loc[panel["decile"] == 10, "car_drift"].mean()
    manual_mean_d1 = panel.loc[panel["decile"] == 1, "car_drift"].mean()
    assert result.loc[0, "spread"] == pytest.approx(manual_mean_d10 - manual_mean_d1)


def test_cost_sensitivity_net_spread_subtracts_round_trip_cost():
    panel = _long_short_panel(
        n_dates=5, events_per_decile_per_date=10, date_shock_scale=0.01, seed=2
    )
    gross_spread = long_short_test(panel).loc[0, "spread"]

    table = cost_sensitivity(panel, costs_bps=(0, 10, 25))

    assert table.loc[table["cost_bps_per_leg"] == 0, "net_spread"].iloc[0] == pytest.approx(
        gross_spread
    )
    # two legs, each a round trip, at 10 bps -> 20 bps = 0.0020 charged
    expected_10bps = gross_spread - 0.0020
    assert table.loc[table["cost_bps_per_leg"] == 10, "net_spread"].iloc[0] == pytest.approx(
        expected_10bps
    )
