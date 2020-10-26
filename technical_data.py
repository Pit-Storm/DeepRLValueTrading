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

endpoint = "technical"
symbol = "AAPL"
exchange = "US"

params = {
    "order": "a",
    "function": "sma"
}
# %%
tmp = ehd.get_data(endpoint=endpoint, symbol=symbol, exchange=exchange, params=params)

# %%
ti_sma_aapl = ehd.make_df(data=tmp, endpoint=endpoint)
