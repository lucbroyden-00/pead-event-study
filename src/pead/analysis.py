"""
CAR/CAAR construction and the long-short decile test for the PEAD event
study.

Five entry points, meant to run in this order:
    build_car_panel(events, abnormal_returns)  -> long per-(event, horizon) CAR panel
    caar_by_decile(panel)                      -> event_time x decile mean-CAR table
    plot_caar_fan(caar, outpath)                -> fan-chart PNG
    long_short_test(panel)                      -> decile 10 - decile 1 spread + t-stats
    cost_sensitivity(panel, costs_bps)          -> net spread after round-trip costs

Event windows follow `docs/methodology.md`: announcement window (0, +1),
drift window (+2, +60).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)

ANNOUNCEMENT_WINDOW_END = 1  # (0, +1)
DRIFT_WINDOW_START = 2
DRIFT_WINDOW_END = 60  # (+2, +60)

RESULTS_DIR = Path("results")
DEFAULT_FAN_CHART_PATH = RESULTS_DIR / "caar_fan.png"


def build_car_panel(events: pd.DataFrame, abnormal_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Merge event-time abnormal returns onto `events` (with its decile
    assignments), producing a long panel of cumulative abnormal returns.

    For each event, `car` at event_time `t` (0 to 60) is the cumulative
    abnormal return summed from horizon 0 through `t`. `car_announcement`
    (the (0, +1) window) and `car_drift` (the (+2, +60) window) are
    derived from the same cumulative series and attached as separate
    columns, constant across every row of a given event -- so both the
    per-horizon fan-chart series and the event-level drift test can be
    read off one panel.

    `abnormal_returns` is expected to be `compute_abnormal_returns`'s
    output (columns: cik, ticker, event_id, event_time, abnormal_return),
    and `events` the decile-assigned events passed to that same call --
    `event_id` is taken from `events` if present, otherwise generated from
    its row position (`events.reset_index(drop=True).index`), which is
    only a valid join key when `events` has the same row order it had
    when `abnormal_returns` was computed.

    Raises if `abnormal_returns` doesn't cover event_time 0 through 60.
    """
    events = events.reset_index(drop=True).copy()
    if "event_id" not in events.columns:
        events = events.assign(event_id=events.index)

    required = {"event_id", "decile", "announcement_date"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events is missing required column(s): {sorted(missing)}")

    long = (
        abnormal_returns.loc[
            abnormal_returns["event_time"].between(0, DRIFT_WINDOW_END),
            ["cik", "ticker", "event_id", "event_time", "abnormal_return"],
        ]
        .sort_values(["event_id", "event_time"])
        .reset_index(drop=True)
    )
    long["car"] = long.groupby("event_id")["abnormal_return"].cumsum()

    car_at = long.pivot(index="event_id", columns="event_time", values="car")
    if ANNOUNCEMENT_WINDOW_END not in car_at.columns or DRIFT_WINDOW_END not in car_at.columns:
        raise ValueError(
            f"abnormal_returns must cover event_time 0 through {DRIFT_WINDOW_END} "
            "to derive the announcement and drift window CARs"
        )
    window_cars = pd.DataFrame(
        {
            "event_id": car_at.index,
            "car_announcement": car_at[ANNOUNCEMENT_WINDOW_END].to_numpy(),
            "car_drift": (car_at[DRIFT_WINDOW_END] - car_at[ANNOUNCEMENT_WINDOW_END]).to_numpy(),
        }
    )

    panel = long.merge(window_cars, on="event_id", how="left")
    panel = panel.merge(
        events[["event_id", "decile", "announcement_date"]], on="event_id", how="left"
    )

    n_unmatched = int(panel["decile"].isna().sum())
    if n_unmatched:
        logger.warning(
            "%d row(s) had no matching event in `events` after the event_id join -- "
            "check that `events` has the same row order used to build `abnormal_returns`",
            n_unmatched,
        )

    return panel[
        [
            "event_id",
            "cik",
            "ticker",
            "announcement_date",
            "decile",
            "event_time",
            "car",
            "car_announcement",
            "car_drift",
        ]
    ]


def caar_by_decile(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Mean CAR by decile at each event time from 0 to 60 (the CAAR fan-chart
    series). Returns a DataFrame indexed by `event_time`, one column per
    decile.
    """
    caar = panel.pivot_table(index="event_time", columns="decile", values="car", aggfunc="mean")
    return caar.sort_index().sort_index(axis=1)


def plot_caar_fan(caar: pd.DataFrame, outpath: str | Path = DEFAULT_FAN_CHART_PATH) -> Path:
    """
    Fan chart: one line per decile, event time on x, CAAR (%) on y.

    Deciles are coloured on a diverging scale (`RdBu_r`) so decile 1 (most
    negative surprise) and decile 10 (most positive) sit at opposite ends
    and read as visually distinct. A vertical dashed line marks +2, the
    start of the drift window. Saved at 150 dpi.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    deciles = sorted(caar.columns)
    n = len(deciles)
    cmap = matplotlib.colormaps["RdBu_r"]

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, decile in enumerate(deciles):
        color = cmap(i / (n - 1)) if n > 1 else cmap(0.5)
        ax.plot(caar.index, caar[decile] * 100, label=f"D{decile}", color=color, linewidth=1.8)

    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axvline(DRIFT_WINDOW_START, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax.annotate(
        f"drift start (+{DRIFT_WINDOW_START})",
        xy=(DRIFT_WINDOW_START, 1),
        xycoords=("data", "axes fraction"),
        xytext=(4, -4),
        textcoords="offset points",
        va="top",
        ha="left",
        fontsize=8,
    )

    ax.set_xlabel("Event time (trading days)")
    ax.set_ylabel("Cumulative abnormal return (%)")
    ax.set_title("CAAR by SUE decile")
    ax.legend(title="Decile", ncol=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    return outpath


def long_short_test(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Decile 10 vs. decile 1 spread in the drift window CAR, with the naive
    and clustered t-statistics reported side by side.

    The spread is the plain difference of group means. Significance comes
    from OLS of drift CAR on `1{decile==10} - 1{decile==1}` (nonzero only
    for the two extreme deciles), fit twice: once with the default
    (i.i.d.) covariance for the "naive" t-stat, once with standard errors
    clustered on `announcement_date` for the "clustered" one. The
    regressor's +1/-1 coding halves the coefficient relative to the raw
    spread, but a t-statistic is invariant to that rescaling, so both
    t-stats test the same hypothesis the raw spread does.

    Returns a one-row summary DataFrame.
    """
    event_level = (
        panel.drop_duplicates(subset="event_id")[
            ["event_id", "decile", "announcement_date", "car_drift"]
        ]
        .dropna(subset=["decile", "car_drift"])
        .copy()
    )

    long_short = event_level.loc[event_level["decile"].isin([1, 10])].copy()
    if long_short.empty:
        raise ValueError("panel has no decile 1 or decile 10 events")

    mean_d10 = long_short.loc[long_short["decile"] == 10, "car_drift"].mean()
    mean_d1 = long_short.loc[long_short["decile"] == 1, "car_drift"].mean()
    spread = mean_d10 - mean_d1

    long_short["indicator"] = np.where(long_short["decile"] == 10, 1.0, -1.0)
    y = long_short["car_drift"].to_numpy()
    X = sm.add_constant(long_short["indicator"].to_numpy())

    naive = sm.OLS(y, X).fit()
    clustered = sm.OLS(y, X).fit(
        cov_type="cluster", cov_kwds={"groups": long_short["announcement_date"]}
    )

    return pd.DataFrame(
        [
            {
                "decile_10_mean": mean_d10,
                "decile_1_mean": mean_d1,
                "spread": spread,
                "naive_t_stat": naive.tvalues[1],
                "clustered_t_stat": clustered.tvalues[1],
                "n_events": len(long_short),
                "n_clusters": long_short["announcement_date"].nunique(),
            }
        ]
    )


def cost_sensitivity(
    panel: pd.DataFrame, costs_bps: tuple[float, ...] = (0, 10, 25)
) -> pd.DataFrame:
    """
    Net long-short drift spread after round-trip transaction costs, at
    each level in `costs_bps`.

    The long-short position is two legs -- long decile 10, short decile 1
    -- each opened and closed (a round trip), so each leg is charged
    `costs_bps` twice; total cost at a given level is `2 * costs_bps`
    (both legs, one round trip each) converted from bps to a return.
    """
    spread = long_short_test(panel).loc[0, "spread"]

    rows = []
    for bps in costs_bps:
        total_cost = 2 * bps / 10_000
        rows.append(
            {
                "cost_bps_per_leg": bps,
                "gross_spread": spread,
                "total_cost": total_cost,
                "net_spread": spread - total_cost,
            }
        )
    return pd.DataFrame(rows)
