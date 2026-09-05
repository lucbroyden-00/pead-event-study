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
- Item 2.02 filings whose reporting lag falls outside **7–75 days** are
  excluded as non-quarterly releases. Example: Apple's 2019-01-02 revenue
  warning, a guidance revision rather than a quarterly result, at a 4-day
  lag.
- The floor was originally 15 days; lowered to 7 after validation against
  JPMorgan showed it reports 12–16 days after quarter end, so a 15-day
  floor rejected roughly 85% of its legitimate quarterly announcements.
  7 days still excludes the Apple guidance-revision case above (4-day
  lag) while accommodating fast reporters.

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

## Earnings expectation model

- **Primary specification: pure seasonal random walk.** `expected_eps =
  eps_lag4q`, so `unexpected_eps = eps - eps_lag4q`, scaled by the share
  price at t-1: `sue = unexpected_eps / price_{t-1}`. Combined with the
  within-quarter winsorising below, this is what `compute_sue` /
  `assign_deciles` produce and what every downstream result is built on.
- **Tested and rejected: seasonal random walk with drift** (Bernard and
  Thomas, 1989): `expected_eps = eps_lag4q + eps_drift`, where `eps_drift`
  is the mean of the last four available year-over-year changes
  (`eps[q] - eps[q-4]`, for each of the four quarters strictly before the
  one being predicted, per company). The motivation was real: a pure
  random walk implicitly assumes no trend in year-over-year earnings
  change, so firms whose earnings changed a lot for entirely predictable
  reasons — a recovery, a decline, an ongoing business transformation —
  can read as "surprised" even when the market saw it coming. In
  practice, though, adding the drift term made the results worse: it
  reduced the long-short drift spread from 0.55% to 0.19% and its
  clustered t-statistic from 1.02 to 0.37, and it degraded the
  monotonicity of the decile sort at the announcement window. The pure
  random walk with winsorising remains the primary specification.
- The drift-term code (`add_earnings_drift`, `eps_drift`, `expected_eps`)
  is kept in the codebase, tested, and importable, but is not called by
  the default pipeline — a documented alternative rather than a deleted
  one, in case the comparison is worth revisiting (e.g. with a different
  drift window, or shrinkage toward zero drift). Rows without at least 3
  of the last 4 year-over-year changes (mostly a company's first few
  quarters in the dataset) are dropped there rather than given an
  unreliable drift estimate; the count is logged.

## SUE winsorisation

- `sue` (the pure-random-walk surprise above) is winsorised within each
  *calendar* quarter at the 1st/99th percentiles before decile
  assignment, producing a `sue_winsorised` column that `assign_deciles`
  ranks on; raw `sue` is kept unchanged for reference. The number of
  observations capped is logged.
- Driven by large one-off writedowns: NRG (SUE ≈ −20.08), PG&E (≈ −13.29),
  Marathon (≈ −14.25), Kraft Heinz (≈ −10.30). These are real GAAP
  figures, not data errors, but poor proxies for earnings *news* — they
  are largely anticipated before the release, non-cash, and concentrated
  in specific quarters and sectors rather than reflecting a surprise the
  market is reacting to.
- Left uncapped, these outliers dominated their calendar quarter's SUE
  distribution and inverted decile 1's drift, which should be negative
  (the market underreacting to bad news) — instead decile 1 was populated
  by writedown-driven observations whose drift ran positive. Winsorising
  within quarter caps their influence on the ranking without discarding
  the observations entirely.

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
