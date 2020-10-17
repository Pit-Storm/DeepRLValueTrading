# %%
###
# Imports
###

from eodhistoricaldata import make_df, get_eod
import pandas as pd
# %%
###
# Reading Data
###

symbol = "AAPL"
exchange = "US"

params = {
    "period": "d",
    "order": "a"
}

# %%
r = get_eod(symbol=symbol, exchange=exchange, params=params)

# %%
eod_aapl = make_df(data = r, endpoint="eod")

# %%
