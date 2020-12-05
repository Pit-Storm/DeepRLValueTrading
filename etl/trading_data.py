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

with open("stocks.txt") as file:
    symb_exchange = file.read().splitlines()

stocks = [item.split(".") for item in symb_exchange]

params = {
    "period": "d",
    "order": "a"
}
# %%
stocks_df = pd.DataFrame()
for stock in stocks:
    tmp = ehd.get_data(endpoint=endpoint, symbol=stock[0], exchange=stock[1], params=params)
    tmp_df = ehd.make_df(data=tmp, endpoint=endpoint)
    tmp_df["symbol"] = stock[0]
    tmp_df = tmp_df.set_index(keys="symbol", append=True)
    stocks_df = stocks_df.append(tmp_df)

stocks_df = (stocks_df.sort_index()
    .loc[("2000-01-03",):("2019-12-31",)]
)
# %%
stocks_df.reset_index().to_csv("../data/stocksdata_trading.csv")
# %%
