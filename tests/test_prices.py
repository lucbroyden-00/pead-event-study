import numpy as np
import pandas as pd
import pytest

from pead.prices import (
    BENCHMARK_TICKER,
    assign_event_day,
    build_trading_calendar,
    compute_abnormal_returns,
)


def _synthetic_prices(n_days: int, tickers=("SPY",), seed: int = 0) -> pd.DataFrame:
    """Business-day price panel with random-walk-ish, always-positive closes."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    data = {}
    for ticker in tickers:
        steps = rng.normal(loc=0.0002, scale=0.01, size=n_days)
        data[ticker] = 100 * np.cumprod(1 + steps)
    return pd.DataFrame(data, index=dates)


def test_after_close_friday_maps_to_following_monday():
    prices = _synthetic_prices(15)  # 2024-01-01 .. 2024-01-19, business days only
    calendar = build_trading_calendar(prices)

    friday = pd.Timestamp("2024-01-05")
    monday = pd.Timestamp("2024-01-08")
    assert friday.day_name() == "Friday"
    assert monday.day_name() == "Monday"

    events = pd.DataFrame(
        {
            "cik": [1],
            "ticker": ["AAPL"],
            "acceptance_et": [pd.Timestamp("2024-01-05 16:30", tz="America/New_York")],
        }
    )

    result = assign_event_day(events, calendar)

    expected_position = calendar.loc[calendar["date"] == monday, "position"].iloc[0]
    assert result.loc[0, "t0_position"] == expected_position


def test_before_open_maps_to_same_trading_day():
    prices = _synthetic_prices(15)
    calendar = build_trading_calendar(prices)

    same_day = pd.Timestamp("2024-01-08")
    events = pd.DataFrame(
        {
            "cik": [1],
            "ticker": ["AAPL"],
            "acceptance_et": [pd.Timestamp("2024-01-08 08:00", tz="America/New_York")],
        }
    )

    result = assign_event_day(events, calendar)

    expected_position = calendar.loc[calendar["date"] == same_day, "position"].iloc[0]
    assert result.loc[0, "t0_position"] == expected_position


def test_intraday_announcement_is_dropped():
    prices = _synthetic_prices(15)
    calendar = build_trading_calendar(prices)

    events = pd.DataFrame(
        {
            "cik": [1],
            "ticker": ["AAPL"],
            "acceptance_et": [pd.Timestamp("2024-01-08 12:00", tz="America/New_York")],
        }
    )

    result = assign_event_day(events, calendar)

    assert len(result) == 0


def test_stock_matching_benchmark_has_zero_abnormal_return_at_every_event_time():
    # End-to-end check on window alignment: if the stock's returns exactly
    # equal the benchmark's every day, every abnormal return must be zero.
    # An off-by-one in the integer indexing would misalign the stock and
    # benchmark windows and break this.
    n_days = 70
    prices = _synthetic_prices(n_days, tickers=(BENCHMARK_TICKER,))
    # STOCK is a scaled multiple of SPY -> identical simple returns every day.
    prices["STOCK"] = prices[BENCHMARK_TICKER] * 3.0

    calendar = build_trading_calendar(prices)

    t0_position = 5  # leaves room for pre=1 .. post=60 within 70 days
    t0_date = calendar.loc[calendar["position"] == t0_position, "date"].iloc[0]

    events = pd.DataFrame(
        {
            "cik": [1],
            "ticker": ["STOCK"],
            "acceptance_et": [pd.Timestamp(t0_date.date(), tz="America/New_York") + pd.Timedelta(hours=8)],
        }
    )
    events = assign_event_day(events, calendar)
    assert events.loc[0, "t0_position"] == t0_position

    abnormal = compute_abnormal_returns(prices, events, pre=1, post=60)

    assert len(abnormal) == 62  # -1 .. +60 inclusive
    assert sorted(abnormal["event_time"]) == list(range(-1, 61))
    assert abnormal["abnormal_return"].abs().max() == pytest.approx(0.0, abs=1e-12)
