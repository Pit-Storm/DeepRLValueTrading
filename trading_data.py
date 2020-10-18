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

# %%
eod_aapl = ehd.make_df(data=tmp, endpoint=endpoint)

# %%
endpoint = "technical"
params.pop("period")
params["function"] = "sma"
# %%
tmp = ehd.get_data(endpoint=endpoint, symbol=symbol, exchange=exchange, params=params)

# %%
ti_sma_aapl = ehd.make_df(data=tmp, endpoint=endpoint)
# %%
endpoint = "fundamentals"
params.pop("function")
# %%
tmp = ehd.get_data(endpoint=endpoint, symbol=symbol, exchange=exchange, params=params)
# %%

# tmp.keys()
# tmp["Financials"]["Balance_Sheet"]["quarterly"]["2020-06-30"].keys()

df_tmp = pd.DataFrame.from_dict(data=tmp["Financials"]["Balance_Sheet"]["quarterly"]["2020-06-30"], orient="index").transpose()
# %%
df_tmp.head()
# %%
