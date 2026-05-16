import numpy as np
import pandas as pd

from src.constants.constants import CANDLE_LINE_COLOR, DEFAULT_TYPE, RALLY, RALLY_CANDLE_BODY_COLOR
from src.model.OHLCColumns import OHLCColumns


class RallyIdentifier:
    def __init__(self):
        # Initialize constants and column names from OHLCColumns model
        pass

    def identify_rally_candles(self, data: pd.DataFrame) -> pd.DataFrame:
        """Method to detect rally in the DataFrame"""

        cols = OHLCColumns()
        # Filter for rally conditions
        # print(data)
        ohlc_fltr = data[cols.OPEN] < data[cols.CLOSE]  # Current open is less than current close
        ohlc_fltr_prv = data[cols.CLOSE] > data[cols.HIGH].shift(
            periods=1
        )  # Current close > previous high
        # Apply the filtering conditions to assign new values to columns

        data[cols.TYPE] = np.where((ohlc_fltr) & (ohlc_fltr_prv), RALLY, DEFAULT_TYPE)
        data[cols.SUBTYPE] = np.where((ohlc_fltr) & (ohlc_fltr_prv), RALLY, DEFAULT_TYPE)
        data[cols.CNDL_BODYCOLOR] = np.where(
            (ohlc_fltr) & (ohlc_fltr_prv), RALLY_CANDLE_BODY_COLOR, DEFAULT_TYPE
        )
        data[cols.CNDL_LLINECOLOR] = np.where(
            (ohlc_fltr) & (ohlc_fltr_prv), CANDLE_LINE_COLOR, DEFAULT_TYPE
        )

        # data[cols.TYPE] = np.where((d_rally) & (ohlc_fltr_prv), RALLY, DEFAULT_TYPE)
        # data[cols.SUBTYPE] = np.where((d_rally) & (ohlc_fltr_prv), RALLY, DEFAULT_TYPE)
        # data[cols.CNDL_BODYCOLOR] = np.where((d_rally) & (ohlc_fltr_prv), RALLY_CANDLE_BODY_COLOR, DEFAULT_TYPE)
        # data[cols.CNDL_LLINECOLOR] = np.where((d_rally) & (ohlc_fltr_prv), CANDLE_LINE_COLOR, DEFAULT_TYPE)

        return data
