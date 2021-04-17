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

with open("stocks.txt") as file:
    symb_exchange = file.read().splitlines()

stocks = [item.split(".") for item in symb_exchange]

params = {
    "period": "d",
    "order": "a",
    "filter": "General::Name"
}

tradeable_stocks = pd.DataFrame(columns=["Name","Index"])

for stock in stocks:
    tmp = []
    tmp.append(ehd.get_data(endpoint=endpoint, symbol=stock[0], exchange=stock[1], params=params))
    if stock[1] != "US":
        tmp.append("EuroStoxx 50")
    else:
        tmp.append("Dow Jones Industrial Average")
    tradeable_stocks.loc[len(tradeable_stocks)+1] = tmp

tradeable_stocks = tradeable_stocks.iloc[tradeable_stocks["Name"].str.lower().argsort()]
tradeable_stocks["#"] = range(1,64)
tradeable_stocks = tradeable_stocks.set_index("#",drop=True)
# %%
tradeable_stocks.to_latex(buf="tradeable_stocks.tex", index_names=False, caption=None, label=None)

# %%
tradeable_stocks.loc[36]
# %%
