import datetime
import vectorbtpro as vbt
import pandas as pd
import numpy as np
import plotly.io as pio
import itertools

from sentiment_indicator import BTCSentIndicator
from get_binance_data import CryptoData

START = '2020-01-01'
END = '2023-08-21'

TICKERS = ["BTCUSDT", 'ETHUSDT', 'XRPUSDT', 'BNBUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'SOLUSDT', 'TRXUSDT', 'LTCUSDT']
TICKERS = ['ETHUSDT']

FEES = 0.15
LEVERAGE = 1
INIT_CASH = 10000
SLIPPAGE = 0.1

STOP_LOSS = 10


def get_cagr_mdd_ratio(pf, c):
    df = pf.stats([
        'total_trades',
        'profit_factor',
        'total_time_exposure',
        'win_rate',
        'avg_winning_trade',
        'avg_losing_trade',
        'avg_winning_trade_duration',
        'avg_losing_trade_duration'
    ], agg_func=None, column=c)

    df["CAGR"] = round(100 * pf.annualized_return, 2)
    df["CAGR/MDD"] = -1 * pf.annualized_return / pf.max_drawdown

    return df


def show_portfolio_plot(pf, c):
    fig = pf.plot(column=c)
    fig.update_layout(
        title=str(c),
        width=1600,
        height=900,
        font=dict(size=14),
    )

    fig.show()


def get_portfolio_stats(pf, c):
    return pf.stats(column=c)


if datetime.datetime.strptime(END, '%Y-%m-%d') > datetime.datetime.today():
    print('END date is in the future')
    exit(1)

if __name__ == '__main__':

    crypto_data = CryptoData.fetch(
        TICKERS,
        start=START,
        end=END)

    crypto_data_conditioned = crypto_data.get('Close').ffill().bfill()

    range = np.arange(0.001, 1, 0.1)
    combinations = list(itertools.product(range, range))
    list1, list2 = [i[0] for i in combinations], [i[1] for i in combinations]

    # optimization phase
    custom_indicator = BTCSentIndicator.run(crypto_data_conditioned, sentiment_cutoff=list1, signal_cutoff=list2)
    #custom_indicator = BTCSentIndicator.run(crypto_data_conditioned, sentiment_cutoff=0.001 , signal_cutoff=0.501)

    entries = custom_indicator.entries
    exits = custom_indicator.exits

    clean_entries, clean_exits = entries.vbt.signals.clean(exits)

    portfolio = vbt.Portfolio.from_signals(
        open=crypto_data.get('Open'),
        high=crypto_data.get('High'),
        close=crypto_data.get('Close'),
        low=crypto_data.get('Low'),
        entries=entries,
        exits=exits,
        fees=FEES / 100,
        freq='D',
        tsl_stop=STOP_LOSS / 100,
        init_cash=INIT_CASH,
        slippage=SLIPPAGE / 100,
        leverage=LEVERAGE,
    )

    columns = portfolio.wrapper.columns
    print_columns = True

    if print_columns:
        for c in columns:
            #show_portfolio_plot(portfolio, c)
            stats = get_portfolio_stats(portfolio, c)

            print(f'{stats}\n\n')
            stats = get_cagr_mdd_ratio(portfolio, c)

            # print(f'{stats}')

df = portfolio.stats([
    'total_trades',
    'profit_factor',
    'total_time_exposure',
    'win_rate',
    'avg_winning_trade',
    'avg_losing_trade',
    'avg_winning_trade_duration',
    'avg_losing_trade_duration'
], agg_func=None)

df["CAGR"] = round(100 * portfolio.annualized_return, 2)
df["CAGR/MDD"] = -1 * portfolio.annualized_return / portfolio.max_drawdown
df = df[df["Total Trades"] > 30]
df = df.sort_values(['Profit Factor'], ascending=[False])

print(f'{df}')
