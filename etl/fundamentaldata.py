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

with open("stocks.txt") as file:
    symb_exchange = file.read().splitlines()

stocks = [item.split(".") for item in symb_exchange]

params = {
    "period": "d",
    "order": "a",
    "filter": "Earnings::History"
}
# %%
eps_df = pd.DataFrame()

for stock in stocks:
    tmp = ehd.get_data(endpoint=endpoint, symbol=stock[0], exchange=stock[1], params=params)
    tmp_df = ehd.make_df(data=tmp, endpoint=endpoint, call_filter=params["filter"])
    tmp_df["symbol"] = stock[0]
    tmp_df = tmp_df.set_index(keys="symbol", append=True)
    eps_df = eps_df.append(tmp_df)

# %%
params["filter"] = "Financials::Balance_Sheet::quarterly"
bv_df = pd.DataFrame()

for stock in stocks:
    tmp = ehd.get_data(endpoint=endpoint, symbol=stock[0], exchange=stock[1], params=params)
    tmp_df = ehd.make_df(data=tmp, endpoint=endpoint, call_filter=params["filter"])
    tmp_df["symbol"] = stock[0]
    tmp_df = tmp_df.set_index(keys="symbol", append=True)
    bv_df = bv_df.append(tmp_df)

# %%
params["filter"] = "outstandingShares::quarterly"
shares_df = pd.DataFrame()

for stock in stocks:
    tmp = ehd.get_data(endpoint=endpoint, symbol=stock[0], exchange=stock[1], params=params)
    tmp_df = ehd.make_df(data=tmp, endpoint=endpoint, call_filter=params["filter"])
    tmp_df["symbol"] = stock[0]
    tmp_df = tmp_df.set_index(keys="symbol", append=True)
    shares_df = shares_df.append(tmp_df)

# %%
stocks_df = eps_df.join(bv_df).join(shares_df)
stocks_df = stocks_df.sort_index()
# %%
dates = stocks_df.index.get_level_values(level="date")
stocks_df = stocks_df[
    (dates >= "2000-01-01") 
    & (dates <= "2019-12-31")
]

stocks_df["shares"] = stocks_df["shares"].fillna(stocks_df["commonStockSharesOutstanding"])

stocks_df = stocks_df.loc[slice(None),("epsActual","book_value","shares")]
# %%
stocks_df.reset_index().to_csv("../data/stocksdata_fundamental.csv")

# stocks_df = stocks_df.fillna(0)
# %%
