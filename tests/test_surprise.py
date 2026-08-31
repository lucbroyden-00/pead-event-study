import numpy as np
import pandas as pd
import pytest

from pead.prices import BENCHMARK_TICKER
from pead.surprise import add_lagged_eps, assign_deciles, compute_sue


def test_add_lagged_eps_does_not_mispair_across_a_missing_quarter():
    # cik=1 is missing 2022-12-31 entirely. A positional .shift(4) would
    # pair 2023-03-31 with 2021-12-31 (four rows back) instead of the
    # true year-ago quarter, 2022-03-31 -- and would pair 2023-12-31 with
    # 2022-03-31 instead of correctly finding no match at all.
    period_ends = pd.to_datetime(
        [
            "2021-12-31",
            "2022-03-31",
            "2022-06-30",
            "2022-09-30",
            # 2022-12-31 missing
            "2023-03-31",
            "2023-06-30",
            "2023-09-30",
            "2023-12-31",
        ]
    )
    eps = pd.DataFrame(
        {
            "cik": 1,
            "period_end": period_ends,
            "eps": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7],
        }
    )

    result = add_lagged_eps(eps)

    q1_2023 = result.loc[result["period_end"] == "2023-03-31"].iloc[0]
    assert q1_2023["eps_lag4q"] == pytest.approx(1.1)  # 2022-03-31, not 2021-12-31 (1.0)

    # 2023-12-31's true year-ago quarter (2022-12-31) doesn't exist, so it
    # must be dropped rather than mispaired with the nearest row that does.
    assert "2023-12-31" not in result["period_end"].astype(str).values


def test_assign_deciles_produces_near_equal_counts_per_quarter():
    rng = np.random.default_rng(0)
    n_obs = 200
    events = pd.DataFrame(
        {
            "sue": rng.normal(size=n_obs),
            "announcement_date": pd.Timestamp("2024-02-15"),  # all in 2024 Q1
        }
    )

    result = assign_deciles(events, n=10)

    counts = result["decile"].value_counts()
    assert set(counts.index) == set(range(1, 11))
    # 200 obs / 10 deciles == 20 each; qcut on continuous data should hit
    # this near-exactly.
    assert counts.min() >= 18
    assert counts.max() <= 22


def test_assign_deciles_drops_thin_quarters():
    events = pd.DataFrame(
        {
            "sue": np.arange(30),
            "announcement_date": pd.Timestamp("2024-02-15"),  # only 30 obs, below the 50 minimum
        }
    )

    result = assign_deciles(events, n=10)

    assert result.empty


def test_compute_sue_matches_formula_directly():
    dates = pd.bdate_range("2024-01-01", periods=10)
    prices = pd.DataFrame(
        {BENCHMARK_TICKER: 100.0, "AAA": [50.0] * 10},
        index=dates,
    )

    events = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "t0_position": [5],
            "eps": [1.5],
            "eps_lag4q": [1.0],
        }
    )

    result = compute_sue(events, prices)

    # price at t-1 (position 4) is 50.0 -> sue = (1.5 - 1.0) / 50.0
    assert result.loc[0, "sue"] == pytest.approx(0.01)


def test_firm_with_zero_surprise_lands_mid_distribution():
    n_firms = 51  # odd count so one firm sits exactly at the median
    dates = pd.bdate_range("2024-01-01", periods=10)
    t0_position = 5

    tickers = [f"F{i}" for i in range(n_firms)]
    price_data = {BENCHMARK_TICKER: 100.0}
    price_data.update({ticker: 100.0 for ticker in tickers})
    prices = pd.DataFrame(price_data, index=dates)

    # eps - eps_lag4q evenly spaced from -1 to +1; the middle firm's
    # surprise is exactly zero.
    diffs = np.linspace(-1.0, 1.0, n_firms)
    zero_idx = n_firms // 2
    assert diffs[zero_idx] == pytest.approx(0.0)

    events = pd.DataFrame(
        {
            "cik": range(n_firms),
            "ticker": tickers,
            "announcement_date": pd.Timestamp("2024-02-15"),  # same calendar quarter
            "t0_position": t0_position,
            "eps": 1.0 + diffs,
            "eps_lag4q": 1.0,
        }
    )

    events = compute_sue(events, prices)
    assert events.loc[zero_idx, "sue"] == pytest.approx(0.0)

    result = assign_deciles(events, n=10)
    zero_firm = result.loc[result["ticker"] == f"F{zero_idx}"].iloc[0]

    # "Mid-distribution": not in the extreme deciles, and rank close to
    # the 50th percentile.
    assert zero_firm["decile"] in (5, 6)
    assert zero_firm["sue_rank"] == pytest.approx(0.5, abs=0.05)
