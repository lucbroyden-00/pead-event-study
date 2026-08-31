# Methodology

Structured record of modelling decisions for this PEAD event study. Read
before changing any modelling code (see `CLAUDE.md`); update in the same
commit as the code change that motivates it.

---

## Data sources

- **SEC EDGAR submissions API** for filing history and **companyfacts API**
  for XBRL financials. Free, no key required, but requires a contact
  `User-Agent` header and is subject to a 10 requests/second limit.
- **yfinance** for prices and split ratios.

## Sample period

Effectively begins 2009, constrained by XBRL phase-in. Pre-2009 filings
would require parsing raw filing text, which is out of scope. One extra
year (2009) arrives via prior-year comparatives inside FY2010 10-Ks.

## Event identification

- The event is an 8-K carrying **Item 2.02** (Results of Operations), not
  the 10-Q filing date — the press release precedes the 10-Q by days or
  weeks, so using the 10-Q date would miss the announcement entirely.
- `t=0` is derived from `acceptanceDateTime`, which EDGAR returns in UTC
  and must be converted to Eastern. Filings accepted at or after 16:00 ET
  have their price reaction on the following trading day.
- Item 2.02 filings whose reporting lag falls outside **15–75 days** are
  excluded as non-quarterly releases. Example: Apple's 2019-01-02 revenue
  warning, a guidance revision rather than a quarterly result.

## EPS handling

- **Earliest-filed value kept for each period**, preserving point-in-time
  correctness. Later appearances are comparatives or restatements and
  would introduce look-ahead bias.
- **Period identity derived from start/end dates, never from `fy`/`fp`.**
  Those fields describe the fiscal period focus of the filing, not the
  fact — prior-year comparatives inherit the current filing's tags, which
  caused derived Q4 values with end dates two years before their start
  dates.
- **Q4 is not separately reported in post-2021 10-Ks** and is derived as
  FY minus Q1+Q2+Q3. Pre-2021, large filers tagged Q4 directly under Item
  302 selected quarterly data, labelled `fp="FY"` despite 90-day
  durations.
- Derived Q4 carries rounding error of one to two cents from accumulated
  per-quarter rounding. Immaterial for surprise calculation.
- **EPS is as-reported and not split-adjusted.** Adjusted using cumulative
  split ratios from yfinance applied to splits occurring after each
  period end. Apple's Q4 EPS falls 8.26 → 1.42 across the 2014 7-for-1
  split and 3.03 → 0.73 across the 2020 4-for-1; unadjusted, a seasonal
  random walk surprise measure would read these as catastrophic misses.

## Event study parameters

- **Estimation window:** 250 trading days ending 21 days before the
  event, the gap avoiding contamination from pre-announcement drift and
  changing volatility.
- **Announcement window:** (0, +1).
- **Drift window:** (+2, +60).

## Validation

Pipeline validated against Apple (CIK 320193, September fiscal year end,
reports after the close) and JPMorgan (CIK 19617, December fiscal year
end, reports before the open). Both produce four quarters per year with
no inverted dates and durations of 89–97 days.

## Known limitations

- Universe constructed from current index membership, introducing
  survivorship bias.
- Apple's 52/53-week fiscal calendar produces occasional 97-day quarters.
- Three Apple announcements occur intraday, where `t=0` cannot be cleanly
  assigned without intraday prices.
