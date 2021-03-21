# %%
###
# Imports
###

import eodhistoricaldata as ehd
import pandas as pd
import numpy as np
from pathlib import Path
# %%
###
# Reading Data
###

stocksdata_trading_fp = Path.cwd().parent.joinpath("data","stocksdata_trading.csv")
stocksdata_funda_fp = Path.cwd().parent.joinpath("data","stocksdata_fundamental.csv")
stocksdata_forex_fp = Path.cwd().parent.joinpath("data","stocksdata_forex.csv")

trading_df = (pd.read_csv(stocksdata_trading_fp, parse_dates=["date"])
    .set_index(["date","symbol"])
    .drop(columns=["Unnamed: 0"])
     .rename(columns={"adjusted_close":"adj_close"})
)
funda_df = (pd.read_csv(stocksdata_funda_fp, parse_dates=["date"])
    .set_index(["date","symbol"])
    .drop(columns=["Unnamed: 0"])
    .rename(columns={"epsActual":"eps", "book_value": "bv", "shares": "shrs"})
)
forex_df = (pd.read_csv(stocksdata_forex_fp, parse_dates=["date"])
    .set_index(["date"])
    .drop(columns=["Unnamed: 0"])
    .rename(columns={"macd":"eurusd_macd"})
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
# days on which a stock hasn't been traded with the value before.
# To know if the stock is tradable we need to
# have a column indicating it.
# To do that, we build a helper df with every symbol conjuncted to every date
# after that we joining the helper df with stocks_df.
# Create a column "tradeable", drop helper column and ffill nan values.

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

# join and drop helper
stocks_df = helper_df.join(stocks_df)
stocks_df = stocks_df.drop(columns=["helper"])
# create tradeable column
tradeable_helper = pd.Series(data=(~stocks_df["open"].isna())*1,name="tradeable")
# And insert it on position 2
stocks_df.insert(loc=2, column="tradeable", value=tradeable_helper)

# Fill open and close with previous values
stocks_df["open"] = stocks_df["open"].fillna(method="ffill")
stocks_df["close"] = stocks_df["close"].fillna(method="ffill")
# and replacing and filling other inf/nan values
stocks_df = stocks_df.replace([np.inf, -np.inf], np.nan)
stocks_df = stocks_df.fillna(0)
# %%
# Append the timespecific data
stocks_df["month"] = stocks_df.index.get_level_values(level="date").month
stocks_df["dayofmonth"] = stocks_df.index.get_level_values(level="date").day
stocks_df["dayofweek"] = stocks_df.index.get_level_values(level="date").dayofweek
# %%
# join forex_df to the data
stocks_df = stocks_df.join(forex_df)
stocks_df = stocks_df.fillna(method="ffill")
# %%
stocksdata_all_fp = Path.cwd().parent.joinpath("data","stocksdata_all.csv")
stocks_df.sort_index().reset_index().to_csv(stocksdata_all_fp)
# %%
