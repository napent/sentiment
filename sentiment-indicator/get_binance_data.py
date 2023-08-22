import pandas as pd
import vectorbtpro as vbt

FETCH = 0

def get_binance_data(ticker, timeframe, start, end):
    binance_data = vbt.BinanceData.fetch(ticker, timeframe=timeframe, start=start, end=end, delay=100)

    binance_data.to_csv(dir_path='data/assets')


def condition_df(df, start, end):
    # trim data to the requested timerange
    mask = (df.index >= start) & (df.index < end)
    df = df.loc[mask]

    # fill missing index dates
    df = df.resample('D').asfreq()

    # add missing dates
    all_dates = pd.date_range(start, end, freq="1D", tz=df.index.tz)
    df = df.reindex(all_dates)

    # forward fill for missing data for data from the end of the original df
    # as well as missing data in the middle of the dataset
    # df = df.ffill()

    # fill potential missing dates from start date to start of the original df
    # df = df.bfill()

    return df


def get_yf_symbol(symbol, period='max', start=None, end=None, **kwargs):

    if FETCH:
        while True:
            try:
                get_binance_data(symbol, '1d', start, end)
                break
            except Exception as e:
                print(e)

    df = pd.read_csv('data/assets/' + symbol + '.csv')

    df['Open time'] = pd.to_datetime(df['Open time'], utc=True)
    df.set_index('Open time', inplace=True)

    df = condition_df(df, start, end)
    return df


class CryptoData(vbt.Data):
    @classmethod
    def fetch_symbol(cls, symbol, **kwargs):
        return get_yf_symbol(symbol, **kwargs)
