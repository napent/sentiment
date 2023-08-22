import numpy as np
import pandas as pd
import vectorbtpro as vbt

from get_data import SentimentData
from dsp import process_low_pass

START = '2020-01-01'
END = '2023-08-21'

INDICATOR = 'ETH-SENT'


def custom_apply_func(value2, sentiment_cutoff, signal_cutoff):
    sentiment_data = SentimentData.fetch(
        INDICATOR,
        start=START,
        end=END, silence_warnings=True).data[INDICATOR]

    sentiment_data['value'] = process_low_pass(sentiment_data['value'], cutoff=sentiment_cutoff)

    # fig = sentiment_data.vbt.plot()
    # fig.update_layout(
    #     title='sentiment',
    #     width=1600,
    #     height=300,
    #     font=dict(size=14),
    # )
    #
    # fig.show()

    r, ohcl_len = value2.shape

    entries = sentiment_data.diff().vbt.crossed_below(0)
    entries = entries.shift(1)
    entries = pd.concat([entries] * ohcl_len, axis=1)
    entries = entries.astype(np.bool_).to_numpy()

    exits = sentiment_data.diff().vbt.crossed_above(0)
    exits = exits.shift(1)
    exits = pd.concat([exits] * ohcl_len, axis=1)
    exits = exits.astype(np.bool_).to_numpy()

    input_data = sentiment_data

    input_data['value'] = process_low_pass(input_data['value'], cutoff=signal_cutoff)
    input_data = input_data > input_data.shift(1)
    input_data = input_data.shift(1)
    input_data.value = input_data.value.astype(bool)
    input_data = input_data.fillna(False).astype(bool)

    input_data = pd.concat([input_data] * ohcl_len, axis=1)

    entries = entries & input_data
    exits = exits & ~input_data

    return entries, exits


BTCSentIndicator = vbt.IF(
    class_name='Sentiment Oscillator',
    short_name='si',
    input_names=['value'],
    param_names=['sentiment_cutoff', 'signal_cutoff'],
    output_names=['entries', 'exits']
).with_apply_func(apply_func=custom_apply_func)
