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

exchange = "US"
symbols = [
    "AAPL"
]
endpoints = [
    "eod", "fundamentals"
]
params = {
    "period": "d",
    "order": "a"
}

frames = {}
# %%
for symbol in symbols:
    for endpoint in endpoints:
        if endpoint == "eod":
            tmp_eod = ehd.get_data(endpoint=endpoint, symbol=symbol, exchange=exchange, params=params)
            tmp_eod_df = ehd.make_df(data=tmp_eod, endpoint=endpoint)
        elif endpoint == "fundamentals":
            # Get EPS
            params["filter"] =  "Earnings::History"
            tmp_eps = ehd.get_data(endpoint=endpoint, symbol=symbol, exchange=exchange, params=params)
            tmp_eps_df = ehd.make_df(data=tmp_eps, endpoint=endpoint, call_filter=params["filter"])
            # Get BV
            params["filter"] = "Financials::Balance_Sheet::quarterly"
            tmp_bv = ehd.get_data(endpoint=endpoint, symbol=symbol, exchange=exchange, params=params)
            tmp_bv_df = ehd.make_df(data=tmp_bv, endpoint=endpoint, call_filter=params["filter"])

    df_stock = tmp_eod_df.join([tmp_eps_df["epsActual"], tmp_bv_df["book_value"]])
    df_stock["p_to_bv"] = df_stock["adjusted_close"] / df_stock["book_value"]
    df_stock["p_to_e"] = df_stock["adjusted_close"] / df_stock["epsActual"]
    df_stock.drop(columns=["book_value"], inplace=True)
    df_stock.rename(columns={"epsActual":"eps","adjusted_close":"adj_close"}, inplace=True)
    df_Stock.fillna(method="ffill", inplace=True)

    frames[symbol] = df_stock.copy()
# %%
frames["AAPL"].describe()
# %%
