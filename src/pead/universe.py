"""
Builds the per-company event dataset that everything else in the pipeline
consumes: one row per (company, earnings announcement), joined to the
fiscal quarter it reports on and that quarter's split-adjusted EPS.

Single entry point:
    build_event_dataset(tickers=None, start="2015-01-01", refresh=False)
        -> (events: pd.DataFrame, failures: pd.DataFrame)

Wraps `pead.edgar` per company (announcements + EPS + split adjustment +
period matching, per `docs/methodology.md`), caches each company's result
to `data/universe/{cik}.parquet` (never committed -- see CLAUDE.md), and
collects rather than raises on a company that fails, so one bad CIK
doesn't abort a run across hundreds of companies.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from pead.edgar import (
    EdgarClient,
    adjust_for_splits,
    get_earnings_announcements,
    get_quarterly_eps,
    match_announcements_to_periods,
)
from pead.prices import get_sp500_constituents

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/universe")

# Item 2.02 filings whose reporting lag falls outside this range are
# guidance revisions / other non-quarterly releases, not the quarterly
# earnings announcement -- see docs/methodology.md ("Event identification").
# Floor is 7, not 15: JPMorgan reports 12-16 days after quarter end, so a
# 15-day floor rejected ~85% of its legitimate quarterly announcements.
MIN_REPORTING_LAG_DAYS = 7
MAX_REPORTING_LAG_DAYS = 75

EVENT_COLUMNS = ["cik", "ticker", "announcement_date", "acceptance_et", "period_end", "eps"]


def _resolve_universe(tickers: list[str] | None) -> pd.DataFrame:
    """Return a (ticker, cik) DataFrame for the requested tickers, or the
    full current S&P 500 if `tickers` is None."""
    if tickers is None:
        constituents = get_sp500_constituents()
        return constituents[["ticker", "cik"]].drop_duplicates().reset_index(drop=True)

    mapping = EdgarClient().ticker_to_cik()
    universe = mapping[mapping["ticker"].isin(tickers)][["ticker", "cik"]].drop_duplicates()

    missing = set(tickers) - set(universe["ticker"])
    if missing:
        logger.warning("Could not find a CIK for %d ticker(s): %s", len(missing), sorted(missing))

    return universe.reset_index(drop=True)


def _build_company_events(client: EdgarClient, cik: int, ticker: str) -> pd.DataFrame:
    """
    One company's announcements joined to the fiscal period each reports
    on and that period's split-adjusted EPS. Raises on any failure --
    the caller is responsible for catching and recording it.
    """
    events = get_earnings_announcements(client, cik)
    eps = get_quarterly_eps(client, cik)
    if eps.empty:
        raise ValueError("no EarningsPerShareDiluted facts in companyfacts")

    eps_adj = adjust_for_splits(eps, ticker)
    matched = match_announcements_to_periods(events, eps_adj, raise_on_bad_lag=False)

    in_lag_window = matched["days_since_period_end"].between(
        MIN_REPORTING_LAG_DAYS, MAX_REPORTING_LAG_DAYS
    )
    matched = matched.loc[in_lag_window].copy()
    matched["ticker"] = ticker

    matched = matched.rename(
        columns={"filing_date": "announcement_date", "end": "period_end", "eps_adj": "eps"}
    )
    return matched[EVENT_COLUMNS].reset_index(drop=True)


def build_event_dataset(
    tickers: list[str] | None = None, start: str = "2015-01-01", refresh: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the long event dataset across the universe: one row per
    (cik, ticker, announcement_date, period_end, eps).

    `tickers=None` (the default) uses the *current* S&P 500 constituents
    from `pead.prices.get_sp500_constituents`, which introduces
    survivorship bias -- see `docs/methodology.md` ("Known limitations").

    Each company's result is cached independently to
    `data/universe/{cik}.parquet` and skipped on subsequent calls unless
    `refresh=True`; the cache holds each company's full history regardless
    of `start`, so widening `start` later doesn't require re-fetching.
    `start` only filters the returned data.

    A company that fails (no XBRL facts, a bad EDGAR response, a matching
    error) is recorded rather than raising, so one bad CIK doesn't abort
    a run across hundreds of companies. Progress is logged every 25
    companies.

    Returns `(events, failures)`. `failures` has columns `ticker`, `cik`,
    `error` -- a per-quarter-of-cik run's worth of failures alongside a
    summary logged as a warning (count and breakdown by error message).
    """
    universe = _resolve_universe(tickers)
    client = EdgarClient()
    start_ts = pd.Timestamp(start)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    failures = []

    for i, row in enumerate(universe.itertuples(index=False), start=1):
        cache_path = CACHE_DIR / f"{row.cik}.parquet"

        if cache_path.exists() and not refresh:
            frames.append(pd.read_parquet(cache_path))
        else:
            try:
                company_events = _build_company_events(client, row.cik, row.ticker)
                company_events.to_parquet(cache_path)
                frames.append(company_events)
            except Exception as exc:  # noqa: BLE001 -- one bad company must not abort the run
                failures.append({"ticker": row.ticker, "cik": row.cik, "error": str(exc)})

        if i % 25 == 0:
            logger.info(
                "Processed %d/%d companies (%d failed so far)", i, len(universe), len(failures)
            )

    events = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=EVENT_COLUMNS)
    )
    events = events.loc[events["announcement_date"] >= start_ts].reset_index(drop=True)

    failures_df = pd.DataFrame(failures, columns=["ticker", "cik", "error"])
    if not failures_df.empty:
        logger.warning(
            "%d/%d companies failed: %s",
            len(failures_df),
            len(universe),
            failures_df["error"].value_counts().to_dict(),
        )

    return events, failures_df
