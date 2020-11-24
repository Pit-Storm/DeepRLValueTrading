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

trading_df = (pd.read_csv("stocksdata_trading.csv", parse_dates=["date"])
    .set_index(["date","symbol"])
    .drop(columns=["Unnamed: 0"])
     .rename(columns={"adjusted_close":"adj_close"})
)
funda_df = (pd.read_csv("stocksdata_fundamental.csv", parse_dates=["date"])
    .set_index(["date","symbol"])
    .drop(columns=["Unnamed: 0"])
    .rename(columns={"epsActual":"eps", "book_value": "bv", "shares": "shrs"})
)
# %%
# Handling trading NaN values
trading_df = trading_df.fillna(0)

# %%
# Handle fundamental data NaN values
symbols = funda_df.index.get_level_values(level="symbol").unique()

for symbol in symbols:
    funda_df.loc[(slice(None),symbol),("bv")] = funda_df.loc[(slice(None),symbol),("bv")].interpolate(limit=4,limit_area="inside")
    funda_df.loc[(slice(None),symbol),("shrs")] = funda_df.loc[(slice(None),symbol),("shrs")].interpolate(limit=4,limit_area="inside")

funda_df = funda_df.fillna(0)
# %%
stocks_df = trading_df.join(funda_df)
# %%
# Handling missing values after joining
# First forward fill the fundamental data, because we assume
# that it doesn't change after it has been released first time
stocks_df = stocks_df.fillna(method="ffill")
# If there hasn't been a value before, fill with 0
stocks_df = stocks_df.fillna(0)
# %%
stocks_df["p_to_bv"] = stocks_df["adj_close"] / stocks_df["bv"]
stocks_df["p_to_e"] = stocks_df["adj_close"] / stocks_df["eps"]
stocks_df.drop(columns=["bv"], inplace=True)

# %%
# Because we got Data from different exchanges
# And they have different trading days, we have to fill the
# days on which a stock hasn't been traded with 0
# To do that, we build a helper df with every symbol conjuncted to every date
# after that we joining the helper df with stocks_df
# and filling the nan values with 0.

helper_df = pd.DataFrame()
dates = stocks_df.index.get_level_values(level="date").unique()
symbols = stocks_df.index.get_level_values(level="symbol").unique()

for symbol in symbols:
    tmp = pd.DataFrame()
    tmp["date"] = dates
    tmp["symbol"] = symbol
    tmp["helper"] = 0
    helper_df = helper_df.append(tmp)

helper_df = helper_df.set_index(["date","symbol"])

stocks_df = helper_df.join(stocks_df)

stocks_df = stocks_df.fillna(0)
stocks_df = stocks_df.drop(columns=["helper"])
# %%
stocks_df.reset_index().to_csv("stocksdata_all.csv")
# %%
