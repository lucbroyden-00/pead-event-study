"""
EDGAR client for earnings announcement dates and as-reported EPS.

Two public entry points:
    get_earnings_announcements(cik)  -> DataFrame of 8-K Item 2.02 events
    get_quarterly_eps(cik)           -> DataFrame of point-in-time quarterly EPS

The SEC requires a descriptive User-Agent and rate-limits to 10 req/sec.
Set EDGAR_UA before use, e.g.
    export EDGAR_UA="Luc Somebody luc@example.com"

NOTE: not validated against the live API. Verify a few known announcement
dates by hand (Apple's are easy to check) before trusting the output.
"""

from __future__ import annotations

import os
import time
import threading
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{filename}"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

CACHE_DIR = Path("data/edgar")
MAX_REQUESTS_PER_SECOND = 8  # SEC limit is 10; leave headroom


class RateLimiter:
    """Blocks so that calls never exceed `per_second` across threads."""

    def __init__(self, per_second: int) -> None:
        self._min_interval = 1.0 / per_second
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()


class EdgarClient:
    def __init__(self, user_agent: str | None = None) -> None:
        ua = user_agent or os.environ.get("EDGAR_UA")
        if not ua or "@" not in ua:
            raise ValueError(
                "Set EDGAR_UA to 'Your Name your@email.com'. "
                "The SEC blocks requests without a contact User-Agent."
            )
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
        self.limiter = RateLimiter(MAX_REQUESTS_PER_SECOND)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def get_json(self, url: str, retries: int = 3) -> dict:
        for attempt in range(retries):
            self.limiter.wait()
            response = self.session.get(url, timeout=30)
            if response.status_code == 200:
                return response.json()
            if response.status_code == 404:
                raise FileNotFoundError(url)
            time.sleep(2**attempt)  # back off on 429 / 5xx
        response.raise_for_status()
        raise RuntimeError(f"Failed after {retries} attempts: {url}")

    def ticker_to_cik(self) -> pd.DataFrame:
        """Current ticker -> CIK mapping. Note: point-in-time only for today."""
        raw = self.get_json(SEC_TICKERS_URL)
        return pd.DataFrame(
            [
                {"ticker": v["ticker"], "cik": int(v["cik_str"]), "name": v["title"]}
                for v in raw.values()
            ]
        )

    def get_all_filings(self, cik: int) -> pd.DataFrame:
        """
        Full filing history, following pagination.

        The submissions JSON holds only the most recent ~1000 filings inline;
        older ones sit in the files listed under filings.files. Skipping those
        silently truncates history, so we always follow them.
        """
        payload = self.get_json(SUBMISSIONS_URL.format(cik=cik))
        frames = [pd.DataFrame(payload["filings"]["recent"])]

        for page in payload["filings"].get("files", []):
            older = self.get_json(SUBMISSIONS_PAGE_URL.format(filename=page["name"]))
            frames.append(pd.DataFrame(older))

        filings = pd.concat(frames, ignore_index=True)
        filings["cik"] = cik
        filings["filingDate"] = pd.to_datetime(filings["filingDate"])
        filings["reportDate"] = pd.to_datetime(filings["reportDate"], errors="coerce")
        return filings


def _parse_acceptance(series: pd.Series) -> pd.Series:
    """
    EDGAR acceptance timestamps are UTC. Localise as UTC, then convert to
    Eastern. Validated against Apple: releases land at 16:30 ET.
    """
    utc = pd.to_datetime(series, errors="coerce", utc=True)
    return utc.dt.tz_convert("America/New_York")


def get_earnings_announcements(client: EdgarClient, cik: int) -> pd.DataFrame:
    """
    8-K filings carrying Item 2.02 (Results of Operations), which is the
    earnings press release.

    Returns columns:
        cik, accession, filing_date, report_date, acceptance_et, after_close
    `after_close` flags filings accepted at/after 16:00 ET, whose price
    reaction lands on the following trading day.
    """
    filings = client.get_all_filings(cik)

    is_8k = filings["form"].str.upper().str.startswith("8-K", na=False)
    has_item_202 = filings["items"].fillna("").str.contains(r"\b2\.02\b", regex=True)
    events = filings.loc[is_8k & has_item_202].copy()

    events["acceptance_et"] = _parse_acceptance(events["acceptanceDateTime"])
    events["after_close"] = events["acceptance_et"].dt.hour >= 16

    events = events.rename(
        columns={"accessionNumber": "accession", "filingDate": "filing_date",
                 "reportDate": "report_date"}
    )
    cols = ["cik", "accession", "filing_date", "report_date", "acceptance_et", "after_close"]
    return events[cols].sort_values("filing_date").reset_index(drop=True)


def _derive_q4(quarterly: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    """
    Q4 EPS is never filed as its own XBRL fact -- a 10-K tags the full year
    as fp="FY", not fp="Q4".

    `fy`/`fp` describe the fiscal period *focus of the filing*, not the
    period a given fact covers: a 10-K carries prior-year comparatives
    stamped with the *current* filing's fy, and fp isn't trustworthy either
    (some rows carry fp="FY" on a 90-day duration). So this uses start/end
    dates only, never fy or fp: for each annual row, a fiscal year is
    "complete" when exactly three quarterly rows nest entirely inside the
    annual period's [start, end], and Q4 = FY - sum(those three). Years
    without exactly three nested quarters are skipped.
    """
    rows = []
    for _, annual_row in annual.iterrows():
        in_year = quarterly[
            (quarterly["start"] >= annual_row["start"]) & (quarterly["end"] <= annual_row["end"])
        ]
        if len(in_year) != 3:
            continue

        rows.append(
            {
                "cik": annual_row["cik"],
                "start": in_year["end"].max() + pd.Timedelta(days=1),
                "end": annual_row["end"],
                "val": annual_row["val"] - in_year["val"].sum(),
                "filed": annual_row["filed"],
                "form": annual_row["form"],
                "fy": annual_row["fy"],
                "fp": "Q4",
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        duration = (result["end"] - result["start"]).dt.days
        assert (result["end"] > result["start"]).all(), "derived Q4 row has end <= start"
        assert duration.between(80, 100).all(), "derived Q4 row has a non-quarterly duration"
    return result


def get_quarterly_eps(client: EdgarClient, cik: int) -> pd.DataFrame:
    """
    As-reported diluted EPS by fiscal quarter, point-in-time.

    Two corrections applied:
      1. Keep only ~quarterly durations (annual figures from the 10-K would
         otherwise be mixed in with quarters).
      2. Where a period appears in several filings, keep the EARLIEST filed
         value. Later appearances are comparatives or restatements; using them
         would leak information that was not available at the time.

    Q4 is NOT present in the raw data (10-Ks tag the full year as fp="FY",
    not Q4), so it is derived as FY minus Q1+Q2+Q3 for each fiscal year
    where all four figures are available. Classification into quarterly vs.
    annual, and the matching of quarters to their fiscal year, is done
    entirely from start/end dates -- fy and fp are not used, since they
    describe the filing's own period focus and can mislabel comparatives
    from a different period. See `_derive_q4`.
    """
    facts = client.get_json(COMPANYFACTS_URL.format(cik=cik))
    concept = facts["facts"]["us-gaap"].get("EarningsPerShareDiluted")
    if concept is None:
        return pd.DataFrame()

    rows = []
    for unit_facts in concept["units"].values():
        rows.extend(unit_facts)

    eps = pd.DataFrame(rows)
    eps["start"] = pd.to_datetime(eps["start"], errors="coerce")
    eps["end"] = pd.to_datetime(eps["end"], errors="coerce")
    eps["filed"] = pd.to_datetime(eps["filed"])
    eps["duration_days"] = (eps["end"] - eps["start"]).dt.days

    quarterly = eps[eps["duration_days"].between(80, 100)].copy()
    quarterly = (
        quarterly.sort_values("filed")
        # dedup on `end` alone: filings disagree on period start by a day
        # or two, so (start, end) treats the same quarter as near-duplicates.
        .drop_duplicates(subset=["end"], keep="first")
        .assign(cik=cik)
    )

    annual = eps[eps["duration_days"].between(350, 380)].copy()
    annual = (
        annual.sort_values("filed")
        # dedup on `end` alone, same reasoning as quarterly above -- and
        # fy isn't a safe dedup key (see `_derive_q4`).
        .drop_duplicates(subset=["end"], keep="first")
        .assign(cik=cik)
    )

    q4 = _derive_q4(quarterly, annual)
    quarterly = pd.concat([quarterly, q4], ignore_index=True)

    # Pre-2021 10-Ks reported Q4 directly (Item 302 selected quarterly
    # data) but tagged it fp="FY" -- the same tag used for the full-year
    # figure itself. fp is otherwise unreliable (see `_derive_q4`), but a
    # quarterly-duration row whose end lands exactly on a fiscal year end
    # is Q4 by definition, so relabel those regardless of what fp says.
    fiscal_year_ends = set(annual["end"])
    quarterly.loc[quarterly["end"].isin(fiscal_year_ends), "fp"] = "Q4"

    cols = ["cik", "start", "end", "val", "filed", "form", "fy", "fp"]
    return quarterly[cols].sort_values("end").reset_index(drop=True)


def adjust_for_splits(eps: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    As-reported XBRL EPS is not split-adjusted: e.g. Apple's Q4 EPS goes
    8.26 (2013) -> 1.42 (2014) across its 7-for-1 split, and 3.03 (2019)
    -> 0.73 (2020) across its 4-for-1, purely from share count with no
    change in underlying earnings power.

    Puts every row on a constant (current) share-count basis by dividing
    `val` by the cumulative product of all splits that occurred *after*
    that row's period `end`. Adds `eps_adj`; `val` is left untouched.
    """
    splits = yf.Ticker(ticker).splits
    if splits.index.tz is not None:
        # yfinance returns split dates tz-aware; `end` is naive. Strip the
        # tz rather than converting -- the calendar date is what matters.
        splits.index = splits.index.tz_localize(None)

    def factor_after(period_end: pd.Timestamp) -> float:
        after = splits[splits.index > period_end]
        return after.prod() if not after.empty else 1.0

    eps = eps.copy()
    eps["eps_adj"] = eps.apply(lambda row: row["val"] / factor_after(row["end"]), axis=1)
    return eps


def match_announcements_to_periods(
    events: pd.DataFrame, eps: pd.DataFrame, raise_on_bad_lag: bool = True
) -> pd.DataFrame:
    """
    Pair each earnings announcement with the fiscal period it reports on:
    the most recent EPS period (per `events`/`eps` `cik`) whose `end` is on
    or before the announcement's `filing_date`, within 75 days.

    Adds `days_since_period_end`. A normal reporting lag is ~7-75 days
    (SEC deadlines run up to 40-90 days depending on filer size, and fast
    reporters like JPMorgan come in at 12-16 days -- a 15-day floor
    rejected roughly 85% of its legitimate quarterly announcements); the
    7-day floor still excludes non-quarterly Item 2.02 filings such as
    guidance revisions (Apple's 2019-01-02 warning had a 4-day lag).
    Anything outside 7-75 days signals a bad pairing rather than a real
    gap, so by default we raise instead of returning silently wrong data.
    Pass `raise_on_bad_lag=False` to get the matches back anyway (with the
    bad rows still flagged in `days_since_period_end`) for diagnostic use.
    """
    left = events.sort_values("filing_date")
    right = eps.sort_values("end")

    matched = pd.merge_asof(
        left,
        right,
        left_on="filing_date",
        right_on="end",
        by="cik",
        direction="backward",
        tolerance=pd.Timedelta(days=75),
    )
    matched["days_since_period_end"] = (matched["filing_date"] - matched["end"]).dt.days
    matched = matched.reset_index(drop=True)

    lag = matched["days_since_period_end"]
    bad = lag.notna() & ~lag.between(7, 75)
    if bad.any() and raise_on_bad_lag:
        raise ValueError(
            f"{bad.sum()} announcement(s) matched a period outside the "
            "7-75 day reporting-lag window -- likely a mismatched pairing."
        )

    return matched
