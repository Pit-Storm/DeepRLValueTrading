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
    "filter": "Financials::Balance_Sheet::quarterly"
}

# %%
tmp = ehd.get_data(endpoint=endpoint, symbol=symbol, exchange=exchange, params=params)
#%%
df_temp = ehd.make_df(data=tmp, endpoint=endpoint, call_filter=params["filter"])
# %%
df_temp.tail()
# %%
