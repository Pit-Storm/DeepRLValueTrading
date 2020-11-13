# %%
###
# Imports
###

import eodhistoricaldata as ehd
import pandas as pd
import seaborn as sns
# %%
###
# Reading Data
###

trading_df = (pd.read_csv("stocksdata_trading.csv", parse_dates=["date"])
    .set_index(["date","symbol"])
    .drop("Unnamed: 0", axis=1)
     .rename(columns={"adjusted_close":"adj_close"})
)
funda_df = (pd.read_csv("stocksdata_fundamental.csv", parse_dates=["date"])
    .set_index(["date","symbol"])
    .drop("Unnamed: 0", axis=1)
    .rename(columns={"epsActual":"eps", "book_value": "bv", "shares": "shrs"})
)
# stocks_df = trading_df.join(funda_df)
# %%
# Handling trading NaN values
trading_df = trading_df.fillna(0)

# %%
# Handle fundamental data NaN values
symbols = funda_df.index.get_level_values(level="symbol")

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
stocks_df.reset_index().to_csv("stocksdata_all.csv")
# %%
