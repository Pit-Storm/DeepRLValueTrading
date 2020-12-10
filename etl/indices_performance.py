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
symbols = ["DJI", "STOXX50E"]
exchange = "INDX"

params = {
    "period": "d",
    "order": "a"
}

# %%
for symbol in symbols:
    tmp = ehd.get_data(endpoint=endpoint, symbol=symbol, exchange=exchange, params=params)
    if symbol == "DJI":
        dow_df = ehd.make_df(data=tmp, endpoint=endpoint, call_filter=symbol)
    if symbol == "STOXX50E":
        stoxx_df = ehd.make_df(data=tmp, endpoint=endpoint, call_filter=symbol)
# %%
dow_df = dow_df.sort_index().loc[("2000-01-03"):("2019-12-31")]
stoxx_df = stoxx_df.sort_index().loc[("2000-01-03"):("2019-12-31")]
# %%
dow_df["symbol"] = "dji"
stoxx_df["symbol"] = "stoxx50e"
# %%
dow_df = dow_df.set_index(keys="symbol", append=True)
stoxx_df = stoxx_df.set_index(keys="symbol", append=True)
# %%
index_df = stoxx_df.append(dow_df)
# %%
index_df = index_df.sort_index()
# %%
index_df.reset_index().to_csv("../data/indices_performance.csv")
# %%
