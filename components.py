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

endpoint = "fundamentals"
exchange = "INDX"

params = {
    "period": "d",
    "order": "a",
    "filter": "Components"
}
# %%
# get actual components of Indices
# DJI = Dow Jones Industrial Average
# STOXX50E = EuroSTOXX 50
# Check below for choosen Stocks from 2020-11-09
symbol = "DJI"
tmp = ehd.get_data(endpoint=endpoint, symbol=symbol, exchange=exchange, params=params)
dji_df = ehd.make_df(data=tmp, endpoint=endpoint, call_filter=params["filter"])

symbol = "STOXX50E"
tmp = ehd.get_data(endpoint=endpoint, symbol=symbol, exchange=exchange, params=params)
stoxx_df = ehd.make_df(data=tmp, endpoint=endpoint, call_filter=params["filter"])

us_eu_stocks = stoxx_df.append(dji_df)
# %%
# Get historical stock data
endpoint = "eod"
_ = params.pop("filter", None)
stocks_data = {}

for row in us_eu_stocks.itertuples():
    tmp = ehd.get_data(endpoint=endpoint, symbol=row.Code, exchange=row.Exchange, params=params)
    print("Received {}.{} data and creating DF...".format(row.Code,row.Exchange))
    stocks_data[row.Code + "." + row.Exchange] = ehd.make_df(data=tmp, endpoint=endpoint)
# %%
# Drop every stock that don'T have data tuple from inclusivly 2000-01-04 and earlier.
# Because we want only Stocks those data no beginning after 2000-01-04
for key, value in stocks_data.items():
    if value.sort_index().loc[:"20000104"].shape[0] == 0:
        _ = stocks_data.pop(key, None)
# %%
# Write used stocks to txt file
# stocks are components of the indices from 2020-11-09
with open('stocks.txt', 'w') as f:
    for stock in stocks_data.keys():
        f.write("%s\n" % stock)
# %%
