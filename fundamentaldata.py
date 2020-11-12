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
# %%
stocks_df = stocks_df.sort_index()
stocks_df = stocks_df[
    (stocks_df.index.get_level_values(level="date") >= "2000-01-01") 
    & (stocks_df.index.get_level_values(level="date") <= "2019-12-31")
]

# %%
# stocks_df = stocks_df.loc[():(),("epsActual","book_value","shares")]
# %%
stocks_df.columns.to_list()
# %%
stocks_df[(stocks_df["shares"].isna()) & (stocks_df["commonStockSharesOutstanding"].isna())].index.get_level_values(level="date").unique().shape
# %%
(stocks_df.index.get_level_values(level="symbol") == "CSCO")
# %%
# TODO: interpolate or fill the missing valuese in "shares" series.
# Maybe with formular: shares = Earnings / EPS