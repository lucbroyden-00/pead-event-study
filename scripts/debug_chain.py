from pead.universe import build_event_dataset
from pead.surprise import add_lagged_eps, compute_sue, assign_deciles
from pead.prices import download_prices, build_trading_calendar, assign_event_day

data, _ = build_event_dataset(start='2015-01-01')
print('1. raw events        :', len(data))
print('   columns:', data.columns.tolist())

tickers = sorted(data['ticker'].unique().tolist())
px = download_prices(tickers, '2014-01-01', '2026-08-31')
cal = build_trading_calendar(px)
print('   price columns     :', px.shape)

e = add_lagged_eps(data)
print('2. after lag join    :', len(e))

e = assign_event_day(e, cal)
print('3. after event day   :', len(e))

e = compute_sue(e, px)
print('4. after compute_sue :', len(e))
if len(e):
    print('   sue nulls:', e['sue'].isna().sum(), 'of', len(e))
    print(e[['ticker','announcement_date','eps','eps_lag4q','sue']].head(5).to_string())

e = assign_deciles(e)
print('5. after deciles     :', len(e))
