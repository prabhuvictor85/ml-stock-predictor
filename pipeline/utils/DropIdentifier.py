import numpy as np
import pandas as pd

from src.constants.constants import CANDLE_LINE_COLOR, DROP, DROP_BASE_CANDLE_BODY_COLOR
from src.model.OHLCColumns import OHLCColumns


class DropIdentifier:
    def __init__(self):
        # Initialize default constants here, no parameters in constructor
        pass

    def identify_drop_candles(self, data: pd.DataFrame) -> pd.DataFrame:
        """Method to identify drop candles in the DataFrame"""
        cols = OHLCColumns()
        # Filter for drop conditions
        ohlc_fltr = data[cols.OPEN] > data[cols.CLOSE]  # Current open > current close
        ohlc_fltr_prv = data[cols.CLOSE] < data[cols.LOW].shift(
            periods=1
        )  # Current close < previous low
        # Apply the filtering conditions to assign new values to columns
        data[cols.TYPE] = np.where((ohlc_fltr) & (ohlc_fltr_prv), DROP, data[cols.TYPE])
        data[cols.SUBTYPE] = np.where((ohlc_fltr) & (ohlc_fltr_prv), DROP, data[cols.SUBTYPE])
        data[cols.CNDL_BODYCOLOR] = np.where(
            (ohlc_fltr) & (ohlc_fltr_prv), DROP_BASE_CANDLE_BODY_COLOR, data[cols.CNDL_BODYCOLOR]
        )
        data[cols.CNDL_LLINECOLOR] = np.where(
            (ohlc_fltr) & (ohlc_fltr_prv), CANDLE_LINE_COLOR, data[cols.CNDL_LLINECOLOR]
        )

        return data
