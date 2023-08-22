import pandas as pd
import requests
import vectorbtpro as vbt
import numpy as np

FETCH = 0


def condition_df(df, start, end):
    # trim data to the requested timerange
    mask = (df.index >= start) & (df.index < end)
    df = df.loc[mask]

    # fill missing index dates
    df = df.resample('D').asfreq()

    # add missing dates
    all_dates = pd.date_range(start, end)
    df = df.reindex(all_dates)

    # forward fill for missing data for data from the end of the original df
    # as well as missing data in the middle of the dataset
    df = df.ffill()

    # fill potential missing dates from start date to start of the original df
    df.fillna(0, inplace=True)

    return df


def get_yf_symbol(symbol, period='max', start=None, end=None, **kwargs):
    if symbol == 'ETH-SENT':
        data = 'data/indexes/dump_enhanced_all-19513_aggregated.json'

        df = pd.read_json(data)
        df.set_index('date', inplace=True)

        df['close'] = df['positive'] * 0.61 + df['negative'] * -1

        df['close'] += 1
        df['close'] *= 50

        df.drop(
            ['items_per_day', 'positive', 'positive_min', 'positive_max', 'negative', 'negative_min', 'negative_max',
             'neutral'], axis=1, inplace=True)

        df.columns = {'value'}

        df = condition_df(df, start, end)

        return df


class SentimentData(vbt.Data):
    @classmethod
    def fetch_symbol(cls, symbol, **kwargs):
        return get_yf_symbol(symbol, **kwargs)
