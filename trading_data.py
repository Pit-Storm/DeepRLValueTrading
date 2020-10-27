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
symbol = "AAPL"
exchange = "US"

params = {
    "period": "d",
    "order": "a"
}

# %%
tmp = ehd.get_data(endpoint=endpoint, symbol=symbol, exchange=exchange, params=params)
stock_aapl = ehd.make_df(data=tmp, endpoint=endpoint)
#%%
stock_aapl.columns
#%%
stock_aapl.tail()
# %%
