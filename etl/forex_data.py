# %%
###
# Imports
###

import eodhistoricaldata as ehd
import pandas as pd
# %%
###
# Reading Data
###

endpoint = "eod"
symbol = "ECBEURUSD"
exchange = "MONEY"

params = {
    "period": "d",
    "order": "a"
}

# %%
tmp = ehd.get_data(endpoint=endpoint, symbol=symbol, exchange=exchange, params=params)
forex_df = ehd.make_df(data=tmp, endpoint=endpoint, call_filter=symbol)
# %%
forex_df = forex_df.sort_index().loc[("2000-01-03"):("2019-12-31")]
# %%
forex_df["macd"].reset_index().to_csv("../data/stocksdata_forex.csv")
# %%
