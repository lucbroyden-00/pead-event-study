"""
Standardized unexpected earnings (SUE) and its cross-sectional deciles.

Three entry points, meant to run in this order:
    add_lagged_eps(eps)          -> eps with a year-ago EPS column attached
    compute_sue(events, prices)  -> events with a sue column
    assign_deciles(events)       -> events with decile / sue_rank columns
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from pead.prices import BENCHMARK_TICKER

logger = logging.getLogger(__name__)

LAG_DAYS = 365
LAG_TOLERANCE_DAYS = 45
MIN_QUARTER_OBSERVATIONS = 50


def add_lagged_eps(
    eps: pd.DataFrame, date_col: str = "period_end", eps_col: str = "eps"
) -> pd.DataFrame:
    """
    Attach each quarter's EPS from four quarters earlier by matching on
    period-end dates ~365 days apart (tolerance +/-45 days), not by
    `.shift(4)`.

    `.shift(4)` assumes an unbroken one-row-per-quarter sequence. When a
    company has a gap (a missed filing, a fiscal-calendar change), the
    row four positions back is no longer the same quarter a year prior --
    shift(4) pairs it anyway, silently. Matching on the actual date
    instead means a gap simply fails to find a match rather than
    fabricating a wrong one.

    Expects `cik` and `date_col` columns; adds `eps_lag4q`. Rows with no
    match within tolerance are dropped (count logged).
    """
    eps = eps.sort_values(date_col).reset_index(drop=True).copy()
    eps["_target_date"] = eps[date_col] - pd.Timedelta(days=LAG_DAYS)

    candidates = (
        eps[["cik", date_col, eps_col]]
        .rename(columns={date_col: "_candidate_date", eps_col: "eps_lag4q"})
        .sort_values("_candidate_date")
    )

    matched = pd.merge_asof(
        eps.sort_values("_target_date"),
        candidates,
        left_on="_target_date",
        right_on="_candidate_date",
        by="cik",
        direction="nearest",
        tolerance=pd.Timedelta(days=LAG_TOLERANCE_DAYS),
    )

    n_missing = int(matched["eps_lag4q"].isna().sum())
    if n_missing:
        logger.info(
            "Dropping %d row(s) with no year-ago EPS match within +/-%d days",
            n_missing,
            LAG_TOLERANCE_DAYS,
        )
    matched = matched.dropna(subset=["eps_lag4q"])

    return (
        matched.drop(columns=["_target_date", "_candidate_date"])
        .sort_values(date_col)
        .reset_index(drop=True)
    )


def compute_sue(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    eps_col: str = "eps",
    eps_lag_col: str = "eps_lag4q",
) -> pd.DataFrame:
    """
    Unexpected earnings: current EPS minus year-ago EPS, scaled by the
    share price at t-1.

    t-1 (not the announcement day or the filing date) is the scale
    because it is the last price set *before* the announcement can move
    it -- scaling by a post-announcement price would divide the surprise
    by a price that already reflects it.

    Expects `events` to carry `ticker`, `t0_position` (see
    `pead.prices.assign_event_day`), `eps_col`, and `eps_lag_col`; `prices`
    must carry `BENCHMARK_TICKER`, whose price dates define the trading
    calendar that `t0_position` indexes into (see
    `pead.prices.build_trading_calendar`). Rows where price or EPS data is
    unavailable get a null `sue` rather than being dropped, since which
    columns are missing on a given row -- the price is out of range or the
    ticker isn't in `prices` -- is worth distinguishing from a real zero
    surprise downstream.

    Adds a `sue` column to `events`.
    """
    if BENCHMARK_TICKER not in prices.columns:
        raise ValueError(f"Benchmark ticker {BENCHMARK_TICKER!r} is missing from prices")

    calendar_dates = prices[BENCHMARK_TICKER].dropna().index.sort_values()
    ordered_prices = prices.loc[calendar_dates]
    n_positions = len(ordered_prices)

    events = events.reset_index(drop=True).copy()
    sue = np.full(len(events), np.nan)

    for i, event in enumerate(events.itertuples(index=False)):
        t0 = event.t0_position
        if pd.isna(t0):
            continue
        t_minus_1 = int(t0) - 1
        if t_minus_1 < 0 or t_minus_1 >= n_positions:
            continue

        ticker = event.ticker
        if ticker not in ordered_prices.columns:
            continue
        price = ordered_prices[ticker].iloc[t_minus_1]

        eps_now = getattr(event, eps_col)
        eps_lag = getattr(event, eps_lag_col)
        if pd.isna(price) or price == 0 or pd.isna(eps_now) or pd.isna(eps_lag):
            continue

        sue[i] = (eps_now - eps_lag) / price

    events["sue"] = sue
    return events


def assign_deciles(
    events: pd.DataFrame, n: int = 10, date_col: str = "announcement_date"
) -> pd.DataFrame:
    """
    Rank `sue` within each *calendar* quarter of `date_col` (not fiscal
    quarter -- companies report on different fiscal calendars, so grouping
    by calendar quarter is what makes the cross-section contemporaneous),
    and cut it into `n` equal-count buckets per quarter.

    Adds `decile` (1 = most negative surprise, ..., n = most positive) and
    `sue_rank` (percentile within the quarter, in [0, 1]). Calendar
    quarters with fewer than `MIN_QUARTER_OBSERVATIONS` (50) events are
    dropped entirely -- deciles are unstable below that -- and logged.
    """
    events = events.copy()
    events["_calendar_quarter"] = pd.PeriodIndex(events[date_col], freq="Q")

    counts = events.groupby("_calendar_quarter")["sue"].count()
    thin_quarters = counts[counts < MIN_QUARTER_OBSERVATIONS].index
    if len(thin_quarters):
        logger.info(
            "Dropping %d calendar quarter(s) with fewer than %d observations: %s",
            len(thin_quarters),
            MIN_QUARTER_OBSERVATIONS,
            sorted(str(q) for q in thin_quarters),
        )
    events = events.loc[~events["_calendar_quarter"].isin(thin_quarters)].copy()

    def _rank_within_quarter(group: pd.DataFrame) -> pd.DataFrame:
        group = group.copy()
        group["sue_rank"] = group["sue"].rank(pct=True, method="average")
        group["decile"] = pd.qcut(group["sue"], n, labels=False, duplicates="drop") + 1
        return group

    events = events.groupby("_calendar_quarter", group_keys=False).apply(_rank_within_quarter)
    return events.drop(columns="_calendar_quarter", errors="ignore").reset_index(drop=True)
