"""
Sanity-check the EDGAR client against Apple (CIK 320193).

Run from the repo root with the venv active:
    export EDGAR_UA="Your Name your@email.com"
    python scripts/validate_edgar.py

This is not a unit test. It prints diagnostics for a human to eyeball,
because the failure modes here are silent: wrong timezone, truncated
history, and misidentified event types all produce plausible-looking
DataFrames that are quietly useless.
"""

import sys

import pandas as pd

from pead.edgar import (
    EdgarClient,
    get_earnings_announcements,
    get_quarterly_eps,
    match_announcements_to_periods,
)

APPLE_CIK = 320193
pd.set_option("display.width", 120)


def rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> int:
    client = EdgarClient()

    # ---------------------------------------------------------------
    # CHECK 1: did pagination work?
    # The submissions JSON holds ~1000 filings inline. Apple has filed
    # far more than that, so a total near 1000 means the paginated
    # older pages were never fetched.
    # ---------------------------------------------------------------
    rule("CHECK 1 - Filing history depth")
    filings = client.get_all_filings(APPLE_CIK)
    print(f"Total filings retrieved : {len(filings):,}")
    print(f"Earliest filing date    : {filings['filingDate'].min().date()}")
    print(f"Latest filing date      : {filings['filingDate'].max().date()}")
    if len(filings) <= 1100:
        print("!! SUSPICIOUS: close to 1000. Pagination may have failed silently.")

    # ---------------------------------------------------------------
    # CHECK 2: do we get roughly four earnings events per year?
    # Any year with fewer than 3 or more than 5 means the Item 2.02
    # filter is catching the wrong things, or missing some.
    # ---------------------------------------------------------------
    rule("CHECK 2 - Announcements per calendar year")
    events = get_earnings_announcements(client, APPLE_CIK)
    print(f"Total Item 2.02 events  : {len(events)}")

    per_year = events.groupby(events["filing_date"].dt.year).size()
    print(per_year.to_string())

    odd_years = per_year[(per_year < 3) | (per_year > 5)]
    if not odd_years.empty:
        print("\n!! Years outside the expected 3-5 range:")
        print(odd_years.to_string())

    # ---------------------------------------------------------------
    # CHECK 3: timezone handling.
    # Apple reports AFTER the close. If after_close is mostly False,
    # the acceptance timestamps are being parsed as UTC rather than
    # Eastern, and every event date will be off by one day.
    # ---------------------------------------------------------------
    rule("CHECK 3 - Announcement timing (Apple reports after the close)")
    share_after_close = events["after_close"].mean()
    print(f"Share flagged after_close : {share_after_close:.1%}")
    print("\nAcceptance hour distribution (ET):")
    print(events["acceptance_et"].dt.hour.value_counts().sort_index().to_string())

    if share_after_close < 0.8:
        print("\n!! TIMEZONE BUG LIKELY. Apple's releases should cluster at 16:30 ET.")
        print("   Hours clustering around 20-21 means UTC is being read as ET.")

    # ---------------------------------------------------------------
    # CHECK 4: eyeball actual dates against reality.
    # Look these up in a news search. If they disagree by a day,
    # everything downstream is corrupted.
    # ---------------------------------------------------------------
    rule("CHECK 4 - Ten most recent events (verify a few by hand)")
    print(events.tail(10).to_string(index=False))

    # ---------------------------------------------------------------
    # CHECK 5: EPS coverage.
    # Expect ~3 quarterly observations per fiscal year, because Q4 is
    # NOT reported as a quarter - it must be derived from the annual
    # figure. If you see 4 per year, something unexpected is going on.
    # ---------------------------------------------------------------
    rule("CHECK 5 - Quarterly EPS observations")
    eps = get_quarterly_eps(client, APPLE_CIK)
    print(f"Quarterly EPS rows      : {len(eps)}")
    if eps.empty:
        print("!! No EPS data returned. Check the XBRL concept name.")
        return 1

    print(f"Date range              : {eps['end'].min().date()} to {eps['end'].max().date()}")
    print("\nObservations per fiscal year (expect ~3, not 4 - Q4 is missing by design):")
    print(eps.groupby("fy").size().tail(10).to_string())
    print("\nMost recent five:")
    print(eps.tail(5).to_string(index=False))

    # ---------------------------------------------------------------
    # CHECK 6: can announcements be matched to reported periods?
    # An 8-K's report_date is the filing's own event date, not the fiscal
    # period end, so exact matching against EPS period ends can never
    # work. Instead, asof-join each announcement backward to the most
    # recent EPS period end within a plausible reporting lag.
    # ---------------------------------------------------------------
    rule("CHECK 6 - Matching announcements to EPS periods (asof join)")
    # raise_on_bad_lag=False: this script's job is to surface anomalies,
    # not die on the first one. match_announcements_to_periods still flags
    # them via days_since_period_end below.
    matched = match_announcements_to_periods(events, eps, raise_on_bad_lag=False)
    n_matched = matched["end"].notna().sum()
    match_rate = n_matched / len(matched) if len(matched) else float("nan")
    print(f"Announcements matched to a period : {n_matched}/{len(matched)} ({match_rate:.1%})")

    lag = matched["days_since_period_end"].dropna()
    print("\ndays_since_period_end - summary stats:")
    print(lag.describe().to_string())
    print("\ndays_since_period_end - percentiles:")
    print(lag.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_string())

    if match_rate < 0.9:
        print("\n!! Low match rate. Check the 75-day tolerance and the fiscal calendar.")

    bad = lag[~lag.between(15, 75)]
    if not bad.empty:
        print(f"\n!! {len(bad)} match(es) outside the 15-75 day reporting-lag window:")
        print(matched.loc[bad.index, ["accession", "filing_date", "end", "days_since_period_end"]]
              .to_string(index=False))
        print("   Likely an 8-K carrying Item 2.02 that isn't a standard quarterly")
        print("   release (e.g. a guidance update), or a period gap in the EPS data.")

    rule("Done - read the output above, do not just check it ran")
    return 0


if __name__ == "__main__":
    sys.exit(main())
