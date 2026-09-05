"""Produce the headline PEAD results: CAAR fan chart, long-short test, cost sensitivity."""

from pead.universe import build_event_dataset
from pead.surprise import add_lagged_eps, compute_sue, assign_deciles
from pead.prices import download_prices, build_trading_calendar, assign_event_day, compute_abnormal_returns
from pead.analysis import build_car_panel, caar_by_decile, plot_caar_fan, long_short_test, cost_sensitivity

data, _ = build_event_dataset(start='2015-01-01')
print('events:', len(data))

tickers = sorted(data['ticker'].unique().tolist())
px = download_prices(tickers, '2014-01-01', '2026-08-31')
cal = build_trading_calendar(px)

events = add_lagged_eps(data)
events = assign_event_day(events, cal)
events = compute_sue(events, px)
events = assign_deciles(events)
print('events with decile:', len(events))

ar = compute_abnormal_returns(px, events)
panel = build_car_panel(events, ar)
print('events in panel:', panel['event_id'].nunique())
print('events per decile:')
print(panel.groupby('decile')['event_id'].nunique().to_string())

caar = caar_by_decile(panel)
print()
print('CAAR by decile at selected horizons (%):')
print((caar.loc[[0, 1, 5, 20, 40, 60]] * 100).round(2).to_string())

plot_caar_fan(caar, 'results/caar_fan.png')
print()
print('chart saved to results/caar_fan.png')

print()
print('long-short test:')
print(long_short_test(panel).to_string())
print()
print('cost sensitivity:')
print(cost_sensitivity(panel).to_string())
