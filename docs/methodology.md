# Methodology

Running log of modelling decisions and the reasoning behind them. Add to
this as we go — short entries, updated in the same commit as the code.

---

**Event = 8-K filing carrying Item 2.02, not the 10-Q filing date.**
The press release (8-K) precedes the 10-Q by days or weeks, so the 10-Q
would put the event window in the wrong place. Item 2.02 is the SEC's
standard tag for an earnings press release.

**t=0 from `acceptanceDateTime`: on/after 16:00 ET → next trading day.**
Markets close at 16:00 ET; a filing accepted at or after that time can't
move the price until the next session, so its event day rolls forward.

**EPS = earliest-filed value per period.**
Later filings restate or repeat the same period as a comparative. Using
anything but the first-filed value would leak information the market
didn't have yet.

**Estimation window: 250 trading days, ending 21 days before the event.**
Standard event-study convention — long enough to estimate normal-return
parameters reliably, with a gap before the event so pre-announcement drift
doesn't contaminate the estimation.

**Announcement window (0, +1); drift window (+2, +60).**
(0, +1) captures the immediate price reaction, including next-day
if the announcement was after close. (+2, +60) is the post-announcement
drift period PEAD studies test for.
