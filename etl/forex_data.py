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
symbol = "ECBEURUSD"
exchange = "MONEY"

params = {
    "period": "d",
    "order": "a"
}

# %%
tmp = ehd.get_data(endpoint=endpoint, symbol=symbol, exchange=exchange, params=params)
tmp_df = ehd.make_df(data=tmp, endpoint=endpoint, call_filter=symbol)

tmp_df.head()

# %%
