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
symbol = "AAPL"
exchange = "US"

params = {
    "period": "d",
    "order": "a",
    "filter": "Earnings::History"
}

# %%
tmp = ehd.get_data(endpoint=endpoint, symbol=symbol, exchange=exchange, params=params)
#%%
df_tmp = ehd.make_df(data=tmp, endpoint=endpoint, call_filter=params["filter"])
# %%
df_tmp.index.dtype
# %%
df_tmp.dtypes

# %%
df_tmp.tail()
# %%
