# base_candle.py

import numpy as np
import pandas as pd

from src.constants.constants import (
    BASE,
    CANDLE_LINE_COLOR,
    DROP,
    DROP_BASE_CANDLE_BODY_COLOR,
    RALLY,
    RALLY_BASE_CANDLE_BODY_COLOR,
)
from src.model.OHLCColumns import OHLCColumns


class BaseCandleIdentifier:
    def __init__(self):
        """
        Initializes the classifier with a dataset.
        :param dataset: Input DataFrame containing OHLC data.
        """
        pass

    def identify_base_candles(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies and classifies base candles in the dataset.
        :return: DataFrame with updated 'Type', 'SubType', and color columns.
        """
        # Bullish Base Filter (Rally Base)
        cols = OHLCColumns()
        rly_bs_fltr = dataset[cols.OPEN] < dataset[cols.CLOSE]
        ohlc_fltr = dataset[cols.CLOSE] <= dataset[cols.HIGH].shift(periods=1)
        ohlc_fltr_prv = dataset[cols.CLOSE] >= dataset[cols.LOW].shift(periods=1)

        # Bearish Base Filter (Drop Base)
        drp_bs_fltr = dataset[cols.CLOSE] < dataset[cols.OPEN]

        # Update DataFrame for Rally Base
        dataset[cols.TYPE] = np.where(
            (rly_bs_fltr) & (ohlc_fltr_prv) & (ohlc_fltr), RALLY, dataset[cols.TYPE]
        )
        dataset[cols.SUBTYPE] = np.where(
            (rly_bs_fltr) & (ohlc_fltr_prv) & (ohlc_fltr), BASE, dataset[cols.SUBTYPE]
        )
        dataset[cols.CNDL_BODYCOLOR] = np.where(
            (rly_bs_fltr) & (ohlc_fltr_prv) & (ohlc_fltr),
            RALLY_BASE_CANDLE_BODY_COLOR,
            dataset[cols.CNDL_BODYCOLOR],
        )
        dataset[cols.CNDL_LLINECOLOR] = np.where(
            (rly_bs_fltr) & (ohlc_fltr_prv) & (ohlc_fltr),
            CANDLE_LINE_COLOR,
            dataset[cols.CNDL_LLINECOLOR],
        )

        # Update DataFrame for Drop Base
        dataset[cols.TYPE] = np.where(
            (drp_bs_fltr) & (ohlc_fltr) & (ohlc_fltr_prv), DROP, dataset[cols.TYPE]
        )
        dataset[cols.SUBTYPE] = np.where(
            (drp_bs_fltr) & (ohlc_fltr) & (ohlc_fltr_prv), BASE, dataset[cols.SUBTYPE]
        )
        dataset[cols.CNDL_BODYCOLOR] = np.where(
            (drp_bs_fltr) & (ohlc_fltr) & (ohlc_fltr_prv),
            DROP_BASE_CANDLE_BODY_COLOR,
            dataset[cols.CNDL_BODYCOLOR],
        )
        dataset[cols.CNDL_LLINECOLOR] = np.where(
            (drp_bs_fltr) & (ohlc_fltr) & (ohlc_fltr_prv),
            CANDLE_LINE_COLOR,
            dataset[cols.CNDL_LLINECOLOR],
        )

        return dataset
