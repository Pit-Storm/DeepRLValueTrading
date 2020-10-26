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
    "order": "a"
}

# %%
tmp = ehd.get_data(endpoint=endpoint, symbol=symbol, exchange=exchange, params=params)
df_tmp = pd.DataFrame.from_dict(data=tmp["Financials"]["Balance_Sheet"]["quarterly"]["2020-06-30"], orient="index").transpose()
# %%
df_tmp.head()
