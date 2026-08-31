import pandas as pd
import pytest

from pead.edgar import (
    _derive_q4,
    _parse_acceptance,
    adjust_for_splits,
    match_announcements_to_periods,
)


def test_parse_acceptance_converts_utc_to_eastern_hour():
    # Apple's releases are accepted at 16:30 ET == 20:30 UTC (EDT, UTC-4).
    raw = pd.Series(["2024-08-01T20:30:00.000Z"])
    result = _parse_acceptance(raw)

    assert result.dt.hour.iloc[0] == 16
    assert result.dt.minute.iloc[0] == 30
    assert str(result.dt.tz) == "America/New_York"


def test_match_announcements_to_periods_pairs_preceding_quarter():
    events = pd.DataFrame(
        {
            "cik": [1],
            "filing_date": pd.to_datetime(["2024-05-02"]),
        }
    )
    eps = pd.DataFrame(
        {
            "cik": [1, 1, 1],
            "end": pd.to_datetime(["2023-12-31", "2024-03-31", "2024-06-30"]),
            "val": [1.0, 1.1, 1.2],
        }
    )

    matched = match_announcements_to_periods(events, eps)

    assert len(matched) == 1
    assert matched.loc[0, "end"] == pd.Timestamp("2024-03-31")
    assert matched.loc[0, "val"] == 1.1
    assert matched.loc[0, "days_since_period_end"] == 32


def _quarters(cik=1, fy=2024):
    return pd.DataFrame(
        {
            "cik": [cik, cik, cik],
            "start": pd.to_datetime(["2024-01-01", "2024-04-01", "2024-07-01"]),
            "end": pd.to_datetime(["2024-03-31", "2024-06-30", "2024-09-29"]),
            "val": [1.0, 1.1, 1.2],
            "filed": pd.to_datetime(["2024-05-01", "2024-08-01", "2024-11-01"]),
            "form": ["10-Q", "10-Q", "10-Q"],
            "fy": [fy, fy, fy],
            "fp": ["Q1", "Q2", "Q3"],
        }
    )


def test_derive_q4_computes_correct_value_from_start_end_nesting():
    quarterly = _quarters()
    annual = pd.DataFrame(
        {
            "cik": [1],
            "start": pd.to_datetime(["2024-01-01"]),
            "end": pd.to_datetime(["2024-12-31"]),
            "val": [4.5],
            "filed": pd.to_datetime(["2025-02-01"]),
            "form": ["10-K"],
            "fy": [2024],
            "fp": ["FY"],
        }
    )

    q4 = _derive_q4(quarterly, annual)

    assert len(q4) == 1
    row = q4.iloc[0]
    assert row["start"] == pd.Timestamp("2024-09-30")
    assert row["end"] == pd.Timestamp("2024-12-31")
    assert row["val"] == pytest.approx(4.5 - (1.0 + 1.1 + 1.2))
    assert row["end"] > row["start"]


def test_derive_q4_ignores_prior_year_comparative_mistagged_with_current_fy():
    # The FY2024 10-K's XBRL facts include a prior-year comparative for
    # FY2023 -- but it's stamped with the *filing's* fy (2024), not the
    # period it actually covers (2023). Both annual rows share fy=2024;
    # only start/end should decide the match, so the comparative (whose
    # dates don't contain the 2024 quarters) must not corrupt the result.
    quarterly = _quarters()
    annual = pd.DataFrame(
        {
            "cik": [1, 1],
            "start": pd.to_datetime(["2023-01-01", "2024-01-01"]),
            "end": pd.to_datetime(["2023-12-31", "2024-12-31"]),
            "val": [4.0, 4.5],
            "filed": pd.to_datetime(["2025-02-01", "2025-02-01"]),
            "form": ["10-K", "10-K"],
            "fy": [2024, 2024],  # comparative mistagged with the current fy
            "fp": ["FY", "FY"],
        }
    )

    q4 = _derive_q4(quarterly, annual)

    assert len(q4) == 1
    row = q4.iloc[0]
    assert row["end"] == pd.Timestamp("2024-12-31")
    assert row["val"] == pytest.approx(4.5 - (1.0 + 1.1 + 1.2))


def test_adjust_for_splits_makes_series_continuous_across_a_split(monkeypatch):
    # Synthetic: true earnings power is flat at 2.00/share in *current*
    # share terms, both before and after a 7-for-1 split. As-reported EPS
    # before the split is stated in pre-split shares, so it's 7x higher.
    eps = pd.DataFrame(
        {
            "cik": [1, 1],
            "end": pd.to_datetime(["2013-09-28", "2014-09-27"]),
            "val": [14.0, 2.0],
        }
    )

    class FakeTicker:
        def __init__(self, ticker):
            self.ticker = ticker

        @property
        def splits(self):
            return pd.Series(
                [7.0],
                index=pd.DatetimeIndex(["2014-06-09"], tz="America/New_York"),
            )

    monkeypatch.setattr("pead.edgar.yf.Ticker", FakeTicker)

    adjusted = adjust_for_splits(eps, "AAPL")

    # Both rows land on the same current-share basis -- no artificial jump.
    assert adjusted.loc[0, "eps_adj"] == pytest.approx(2.0)
    assert adjusted.loc[1, "eps_adj"] == pytest.approx(2.0)
    # raw val is untouched
    assert adjusted.loc[0, "val"] == 14.0
    assert adjusted.loc[1, "val"] == 2.0
