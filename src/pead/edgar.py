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
    EDGAR acceptance timestamps are Eastern time despite sometimes carrying a
    'Z'. Parse naive, then localise to Eastern.
    """
    naive = pd.to_datetime(series, errors="coerce").dt.tz_localize(None)
    return naive.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")


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
    as fp="FY", not fp="Q4". Derive it as FY minus Q1+Q2+Q3 for each fiscal
    year where all three quarters and the FY figure are available.

    Uses the earliest-filed value for each (fy, fp) pair, same point-in-time
    rule as the quarters themselves, and stamps the derived row with the
    FY filing's `filed` date since that is the point at which Q4 first
    becomes computable.
    """
    q123 = quarterly[quarterly["fp"].isin(["Q1", "Q2", "Q3"])]
    q123 = q123.sort_values("filed").drop_duplicates(subset=["fy", "fp"], keep="first")

    complete_years = q123.groupby("fy").size()
    complete_years = complete_years[complete_years == 3].index

    rows = []
    for fy in complete_years:
        fy_annual = annual[annual["fy"] == fy]
        if fy_annual.empty:
            continue
        annual_row = fy_annual.iloc[0]

        year_quarters = q123[q123["fy"] == fy]
        q3 = year_quarters.loc[year_quarters["fp"] == "Q3"].iloc[0]

        rows.append(
            {
                "cik": annual_row["cik"],
                "start": q3["end"],
                "end": annual_row["end"],
                "val": annual_row["val"] - year_quarters["val"].sum(),
                "filed": annual_row["filed"],
                "form": annual_row["form"],
                "fy": fy,
                "fp": "Q4",
            }
        )
    return pd.DataFrame(rows)


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
    where all four figures are available. See `_derive_q4`.
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
        .drop_duplicates(subset=["start", "end"], keep="first")
        .assign(cik=cik)
    )

    annual = eps[(eps["duration_days"].between(350, 380)) & (eps["fp"] == "FY")].copy()
    annual = (
        annual.sort_values("filed")
        .drop_duplicates(subset=["fy"], keep="first")
        .assign(cik=cik)
    )

    q4 = _derive_q4(quarterly, annual)
    quarterly = pd.concat([quarterly, q4], ignore_index=True)

    cols = ["cik", "start", "end", "val", "filed", "form", "fy", "fp"]
    return quarterly[cols].sort_values("end").reset_index(drop=True)
