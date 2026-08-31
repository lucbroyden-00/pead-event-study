"""
Price data and event-window mechanics for the PEAD event study.

Everything here revolves around ``BENCHMARK_TICKER`` (SPY), which does two
jobs at once:

  1. It is the market proxy subtracted from raw returns to get abnormal
     returns.
  2. Its set of price dates *is* the trading calendar. A "trading day" is
     defined as any day SPY has a price -- see `build_trading_calendar`.

Because both jobs share one series, SPY must always be fetched alongside
the universe, and every function here that consumes prices checks for it
and raises a clear error if it is missing rather than silently producing a
calendar (or abnormal returns) that don't mean what they claim to.

Four entry points:
    get_sp500_constituents()                    -> ticker/name/cik universe
    download_prices(tickers, start, end)         -> wide price DataFrame
    build_trading_calendar(prices)               -> date <-> int position
    assign_event_day(events, calendar)           -> events with t0_position
    compute_abnormal_returns(prices, events)     -> long abnormal-return panel
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

BENCHMARK_TICKER = "SPY"

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
PRICE_CACHE_PATH = Path("data/prices.parquet")


def _require_benchmark(prices: pd.DataFrame) -> None:
    if BENCHMARK_TICKER not in prices.columns:
        raise ValueError(
            f"Benchmark ticker {BENCHMARK_TICKER!r} is missing from prices -- it "
            "is required both as the market return and as the trading calendar."
        )


def get_sp500_constituents() -> pd.DataFrame:
    """
    Scrape *current* S&P 500 constituents (ticker, name, CIK) from Wikipedia.

    NOTE: this is current index membership, not historical membership at
    each event date. Companies that left the index between 2009 and today
    (delisted, acquired, dropped for poor performance) are silently
    excluded, which introduces survivorship bias into the universe -- see
    `docs/methodology.md`.
    """
    response = requests.get(
        WIKI_SP500_URL,
        headers={"User-Agent": "Mozilla/5.0 (compatible; pead-event-study research bot)"},
        timeout=30,
    )
    response.raise_for_status()

    table = pd.read_html(response.text, attrs={"id": "constituents"})[0]
    table = table.rename(columns={"Symbol": "ticker", "Security": "name", "CIK": "cik"})
    # yfinance/most price feeds use a hyphen for the share-class separator
    # (BRK-B), Wikipedia uses a dot (BRK.B).
    table["ticker"] = table["ticker"].str.replace(".", "-", regex=False)
    table["cik"] = table["cik"].astype(int)
    return table[["ticker", "name", "cik"]].reset_index(drop=True)


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    refresh: bool = False,
    batch_size: int = 50,
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Bulk-download adjusted daily closes via yfinance.

    `BENCHMARK_TICKER` is always fetched alongside `tickers`, whether or
    not the caller included it. Downloads happen in batches of `batch_size`
    tickers, each retried up to `max_retries` times with exponential
    backoff. Tickers that still have no data after retries are logged and
    dropped rather than failing the whole call.

    Cached to `data/prices.parquet` (never committed -- see CLAUDE.md) and
    loaded from cache unless `refresh=True`.

    Returns a wide DataFrame indexed by date, one column per ticker (the
    ticker's adjusted close, since `auto_adjust=True`). Raises if the
    benchmark itself could not be downloaded.
    """
    if PRICE_CACHE_PATH.exists() and not refresh:
        logger.info("Loading cached prices from %s", PRICE_CACHE_PATH)
        prices = pd.read_parquet(PRICE_CACHE_PATH)
        _require_benchmark(prices)
        return prices

    all_tickers = sorted(set(tickers) | {BENCHMARK_TICKER})

    series: dict[str, pd.Series] = {}
    failed: list[str] = []

    for batch in _chunks(all_tickers, batch_size):
        data = None
        for attempt in range(max_retries):
            try:
                data = yf.download(
                    batch,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    group_by="ticker",
                    threads=True,
                )
                break
            except Exception as exc:  # noqa: BLE001 -- yfinance raises a mix of exception types
                logger.warning(
                    "Batch download failed (attempt %d/%d): %s", attempt + 1, max_retries, exc
                )
                time.sleep(2**attempt)

        if data is None:
            failed.extend(batch)
            continue

        for ticker in batch:
            try:
                closes = data[ticker]["Close"] if len(batch) > 1 else data["Close"]
            except KeyError:
                closes = None
            if closes is None or closes.dropna().empty:
                failed.append(ticker)
                continue
            series[ticker] = closes

    if failed:
        logger.warning("Failed to download prices for %d ticker(s): %s", len(failed), failed)

    if BENCHMARK_TICKER not in series:
        raise RuntimeError(
            f"Benchmark ticker {BENCHMARK_TICKER!r} could not be downloaded -- it is "
            "required for both abnormal returns and the trading calendar."
        )

    prices = pd.DataFrame(series).sort_index()

    PRICE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(PRICE_CACHE_PATH)
    return prices


def build_trading_calendar(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Map each date the benchmark has a price to an integer position (0, 1,
    2, ...) in trading-day order.

    All event-window arithmetic downstream (`assign_event_day`,
    `compute_abnormal_returns`) works in these integer positions, not
    calendar dates. That is what removes the need for holiday and weekend
    handling: "t plus one trading day" is just "position plus one", and it
    is automatically correct because the position sequence already skips
    every day the benchmark didn't trade.

    Returns a DataFrame with columns `date` (the calendar date) and
    `position` (its integer index).
    """
    _require_benchmark(prices)
    dates = prices[BENCHMARK_TICKER].dropna().index.sort_values()
    return pd.DataFrame({"date": pd.DatetimeIndex(dates), "position": range(len(dates))})


def assign_event_day(events: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """
    Map each announcement's Eastern-time acceptance timestamp to its t=0
    integer position in `calendar`.

    Rule (mirrors `docs/methodology.md`):
      - accepted before 09:30 ET  -> same trading day
      - accepted at/after 16:00 ET -> next trading day
      - accepted 09:30-16:00 ET   -> intraday; t=0 cannot be cleanly
        assigned without intraday prices, so the event is dropped (count
        logged).

    Expects an `acceptance_et` column of tz-aware Eastern timestamps (see
    `pead.edgar._parse_acceptance`). Returns `events` with a `t0_position`
    (nullable Int64) column added; rows for events that fall outside the
    calendar entirely (e.g. after-close on the last available trading day)
    get a null position and a logged warning, rather than being silently
    dropped.
    """
    events = events.reset_index(drop=True).copy()

    acceptance = events["acceptance_et"]
    minutes = acceptance.dt.hour * 60 + acceptance.dt.minute
    before_open = minutes < 9 * 60 + 30
    after_close = minutes >= 16 * 60
    intraday = ~(before_open | after_close)

    n_intraday = int(intraday.sum())
    if n_intraday:
        logger.info(
            "Dropping %d intraday announcement(s): t=0 cannot be cleanly "
            "assigned without intraday prices",
            n_intraday,
        )
    events = events.loc[~intraday].copy()
    before_open = before_open.loc[events.index]

    event_date = events["acceptance_et"].dt.tz_localize(None).dt.normalize().to_numpy()
    cal = calendar.sort_values("position")
    cal_dates = cal["date"].dt.normalize().to_numpy()
    cal_positions = cal["position"].to_numpy()

    # Before-open -> same trading day: exact match against the calendar.
    same_day_idx = np.clip(np.searchsorted(cal_dates, event_date), 0, len(cal_dates) - 1)
    same_day_hit = cal_dates[same_day_idx] == event_date

    # After-close -> next trading day: first calendar date strictly after
    # the announcement date (side="right" skips past an exact match too,
    # which matters if the announcement date itself happens to be a
    # trading day).
    next_day_idx = np.searchsorted(cal_dates, event_date, side="right")
    next_day_hit = next_day_idx < len(cal_dates)

    positions = np.full(len(events), np.nan)
    use_same = before_open.to_numpy() & same_day_hit
    positions[use_same] = cal_positions[same_day_idx[use_same]]

    use_next = (~before_open.to_numpy()) & next_day_hit
    positions[use_next] = cal_positions[next_day_idx[use_next]]

    unresolved = np.isnan(positions)
    if unresolved.any():
        logger.warning(
            "%d event(s) fell outside the trading calendar and got no t0_position",
            int(unresolved.sum()),
        )

    events["t0_position"] = pd.array(positions, dtype="Int64")
    return events


def compute_abnormal_returns(
    prices: pd.DataFrame, events: pd.DataFrame, pre: int = 1, post: int = 60
) -> pd.DataFrame:
    """
    For each event, extract market-adjusted abnormal returns from t-`pre`
    to t+`post` using integer calendar positions.

    Expects `events` to carry `cik`, `ticker`, and `t0_position` columns
    (the output of `assign_event_day`); an `event_id` column is used if
    present, otherwise one is generated from the row position.

    The full (-1, +60) range is extracted for every event regardless of
    which analysis window is used downstream: t-1 supplies the price that
    scales SUE and the base for the day-0 return; (0, +1) is the
    announcement window; (+2, +60) is the drift window; (0, +60) is the
    full path for the fan chart. Extract once here, slice later -- don't
    recompute per window.

    Any event with a missing day anywhere in its window is dropped (count
    logged) rather than forward-filled: forward-filling a halted or
    not-yet-listed day would fabricate a zero-abnormal-return day and bias
    CAARs toward zero.

    Returns a long DataFrame with columns: cik, ticker, event_id,
    event_time (int, -pre to +post), abnormal_return.
    """
    _require_benchmark(prices)

    # Restrict to the trading-calendar dates (where the benchmark has a
    # price) so that integer row position == calendar position, matching
    # what `build_trading_calendar` / `assign_event_day` assigned.
    calendar_dates = prices[BENCHMARK_TICKER].dropna().index.sort_values()
    trading_returns = prices.loc[calendar_dates].pct_change()
    benchmark_returns = trading_returns[BENCHMARK_TICKER]
    n_positions = len(trading_returns)

    events = events.reset_index(drop=True)
    if "event_id" not in events.columns:
        events = events.assign(event_id=events.index)

    rows = []
    dropped = 0
    for event in events.itertuples(index=False):
        t0 = event.t0_position
        if pd.isna(t0):
            dropped += 1
            continue
        t0 = int(t0)
        lo, hi = t0 - pre, t0 + post
        if lo < 0 or hi >= n_positions:
            dropped += 1
            continue
        if event.ticker not in trading_returns.columns:
            dropped += 1
            continue

        stock_window = trading_returns[event.ticker].iloc[lo : hi + 1]
        bench_window = benchmark_returns.iloc[lo : hi + 1]
        if stock_window.isna().any() or bench_window.isna().any():
            dropped += 1
            continue

        abnormal = stock_window.to_numpy() - bench_window.to_numpy()
        for event_time, abnormal_return in zip(range(-pre, post + 1), abnormal):
            rows.append(
                {
                    "cik": event.cik,
                    "ticker": event.ticker,
                    "event_id": event.event_id,
                    "event_time": event_time,
                    "abnormal_return": abnormal_return,
                }
            )

    if dropped:
        logger.info(
            "Dropped %d event(s) with a missing day in the (-%d, +%d) window", dropped, pre, post
        )

    return pd.DataFrame(rows, columns=["cik", "ticker", "event_id", "event_time", "abnormal_return"])
