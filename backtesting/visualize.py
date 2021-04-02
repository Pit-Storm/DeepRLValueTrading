# %%
import sys
sys.path.append("..")
from data import handling as dth
import pandas as pd
import numpy as np
from pathlib import Path
%matplotlib inline
import quantstats as qs
import bar_chart_race as bcr
import seaborn as sns
# extend pandas functionality with metrics, etc.
qs.extend_pandas()
# %%


# Get tested dates out of stocksdata
stocksdata_fp = Path.cwd().parent / "data" / "stocksdata_all.csv"
stocksdata_all_df = dth.load_data(stocksdata_fp)
*_, stocksdata_df = dth.train_val_test_split(stocksdata_all_df)
dates = stocksdata_df.index.get_level_values(level="date").unique().tolist()
symbols = stocksdata_df.index.get_level_values(level="symbol").unique().tolist()
stockprices_ser = stocksdata_df["close"].copy()

###
# Generate Multiindex DF for all test results over all experiments over all algos
results_dir = Path.cwd().parent / "results"
tests = []
exp_args = []
# holiday_fix_model = True

algo_dirs = [algo_dir for algo_dir in results_dir.iterdir() if algo_dir.is_dir()]
for algo_dir in algo_dirs:
    exp_dirs = [exp_dir for exp_dir in algo_dir.iterdir() if exp_dir.is_dir()]
    for exp_idx, exp_dir in enumerate(exp_dirs):
        exp_args.append(pd.read_json(path_or_buf=exp_dir.joinpath("run_args.json"), typ="ser"))
        test_files = [test_file for test_file in exp_dir.rglob("env_info/test/*.json") if test_file.is_file()]
        for test_idx, test_file in enumerate(test_files):
            test_file_df = pd.read_json(test_file)
            data = {
                "algo": algo_dir.name,
                "exp": exp_idx,
                "test": test_idx,
                "date": dates,
                "totalValues": test_file_df["totalValues"],
                "cashes": test_file_df["cashes"],
                "numShares": test_file_df["numShares"]
                }
            tests.append(pd.DataFrame(data=data))

# concatenate all tests to one df
tests_df = pd.concat(tests).set_index(["algo", "exp", "test", "date"])
# and all exp_args to one df
exp_args_df = pd.DataFrame(exp_args)
exp_args_df["exp_idx"] = exp_args_df.groupby(by="algo").cumcount()
exp_args_df = exp_args_df.set_index(["algo","exp_idx"])

# Get Sharpe ratio for all tests
tests_sharpe = tests_df["totalValues"].groupby(level=["algo", "exp", "test"]).apply(qs.stats.sharpe).rename("sharpe")
# Get mean sharpe over the tests for each run
tests_sharpe_mean = tests_sharpe.groupby(level=["algo", "exp"]).mean()
# save best exp idx to a pandas series
best_exp_idx = pd.Series(dict(tests_sharpe_mean.groupby(level=["algo"]).idxmax().values.tolist())).rename("best_exp_idx")

# Inside the best experiment, calculate the mean totalValue by date over all tests.
# for that we have to slice out the best exp out of our overall df
best_exp = [tests_df["totalValues"].loc[(algo_name, exp_idx, slice(None), slice(None))] for algo_name, exp_idx in best_exp_idx.items()]
best_exp_df = pd.concat(best_exp).reset_index(level="exp", drop=True).reorder_levels(["algo","date","test"])

# Now we getting our data to make graphs and other metrics.
best_exp_mean_df = best_exp_df.groupby(level=["algo","date"]).mean().unstack(level="algo")

###
# Creating the best exps_args_df to show it later
best_exp_args = [exp_args_df.loc[(algo_name, exp_idx), slice(None)] for algo_name, exp_idx in best_exp_idx.items()]
best_exp_args_df = pd.DataFrame(best_exp_args).reset_index()
best_exp_args_df[['algo','exp_idx']] = pd.DataFrame(best_exp_args_df["index"].tolist(), index= best_exp_args_df.index)
best_exp_args_df = best_exp_args_df.drop(columns="index").set_index(["algo","exp_idx"])

###
### Create indices series
indices_fp = Path.cwd().parent / "data" / "indices_performance.csv"
indices_df = dth.load_data(indices_fp)
*_, indices_df = dth.train_val_test_split(indices_df)

stoxx50e_ser = indices_df[indices_df.index.get_level_values("symbol") == "stoxx50e"]["adjusted_close"]
stoxx50e_ser.index = stoxx50e_ser.index.droplevel(1)
stoxx50e_ser = stoxx50e_ser.pct_change()
stoxx50e_ser = stoxx50e_ser.fillna(0)
stoxx50e_ser = stoxx50e_ser.rename(index="dly_ret").rename_axis("Date")
stoxx50e_ser.name = "stoxx50e"

dji_ser =  indices_df[indices_df.index.get_level_values("symbol") == "dji"]["adjusted_close"]
dji_ser.index = dji_ser.index.droplevel(1)
dji_ser = dji_ser.pct_change()
dji_ser = dji_ser.fillna(0)
dji_ser = dji_ser.rename(index="dly_ret").rename_axis("Date")
dji_ser.name = "dji"

# Create a ETF portfolio
etf_df = pd.DataFrame(index=dates)
etf_df = etf_df.join([dji_ser, stoxx50e_ser]).fillna(0)
# Calculate the weighted daily returns of our ETF portfolio
etf_df["portf_ret"] = etf_df["dji"]*(1-(36/63)) + etf_df["stoxx50e"]*(36/63)
# Calculate the portfolio value
etf_df["totalValue"] = etf_df["portf_ret"].add(1).cumprod().mul(1e6)

etf_ser = etf_df["totalValue"]
etf_ser.name = "ETF"

###
# Create a DF with all portfolio values
# This is the DF to work with
###
portfolios_df = best_exp_mean_df.join(etf_ser)

# %%
###
# Now coming to actual plotting
drl_algos = ["DDPG","PPO","A2C"]
basic_algos = ["BUYHOLD","RANDOM"]

# Matrix of Mean Sharpes (Experiment times Algorithms)
mean_sharpes_df = tests_sharpe_mean.unstack(level=["algo"])
# Color Gradient
color_gradient = sns.light_palette("green", as_cmap=True)
# Apply color gradient to columns of mean_sharpes_df
mean_sharpes_df[drl_algos].T.style.background_gradient(cmap=color_gradient, axis=1).set_precision(2)

# %%
# Showing the variant args of the best experiments
variant_args = ["cagr", "episodic","num_stacks", "trainsampling"]
best_exp_args_df[variant_args].loc[drl_algos]
# %%
# Table for evaluation of best experiments for algos and Bench with:
    # Comp. overall Return
    # Comp. annual growth rate
    # Sharpe ratio
    # Volatility (Standard deviation p.a.)
metrics_df = pd.DataFrame(index=["Total return (%)", "CAGR (%)", "Sharpe ratio", "Std. Dev. (%)"])

for column in portfolios_df.columns:
    sharpe = qs.stats.sharpe(qs.utils.to_returns(portfolios_df[column]))
    ret = qs.stats.comp(qs.utils.to_returns(portfolios_df[column]))
    cagr = qs.stats.cagr(qs.utils.to_returns(portfolios_df[column]))
    vol = qs.stats.volatility(qs.utils.to_returns(portfolios_df[column]), annualize=False)
    metrics_df[column] = [ret*100, cagr*100, sharpe, vol*100]

# Higlight best algorithm
show_portfolios = ["DDPG","PPO","BUYHOLD","A2C","RANDOM","ETF"]
metrics_df[show_portfolios].style.background_gradient(cmap=color_gradient, axis=1).set_precision(3)

###
# Random is better than ETF?
# This is because of the survivor bias.
# We choosed only stocks that lived from 2000 to end 2019
# but the ETF contains loosers.

# %%
# ColorBlind/friendly colormap from https://gist.github.com/thriveth/8560036
colors =    ['#377eb8', '#ff7f00', '#4daf4a',
            '#f781bf', '#a65628', '#984ea3',
            '#999999', '#e41a1c', '#dede00']
drl_colors = colors[0:2]+[colors[3]]

# Plot a linechart for all portfolios resampled to monthly mean value.
portfolios_df[show_portfolios].plot(title="Total portfolio value", legend=True, xlabel="Date", ylabel="Total Value", color=colors).figure.savefig("img/portfolios_total_value.pdf")
# %%
# Bar chart race for Portfolio values
bcr.bar_chart_race(df=portfolios_df.resample("2W").mean(), dpi=330, cmap=[colors[1]]+[colors[0]]+colors[2:],
                filename="vids/best_exp_mean_totalValues_race.mp4", orientation="v",
                fixed_order=True, period_length=1000, interpolate_period=True,
                fixed_max=True, steps_per_period=7,
                title="Portfolio Values over Time")

# %%
########### INVESTIGATIONS

#### Why DDPG wasn't effected by the drop in december 2018?
# The news can tell us, that in december 2018 the US stock market had a historical drop
# Let's check this by plotting the reshaped performance of DJIA and EuroStoxx50
etf_df[["dji", "stoxx50e"]].add(1).cumprod().plot(title="Trend of Indices", ylabel="Cum. Percentage", xlabel="Date", color=colors[-2:]).figure.savefig("img/indices_performance.pdf")
# We can see a drop in DJIA, but also in EuroStoxx50...
# %%
# Did it hold more or less cash during the period?

# Get the Cashes of the Algorithms
best_exp_cashes = [tests_df["cashes"].loc[(algo_name, exp_idx, slice(None), slice(None))] for algo_name, exp_idx in best_exp_idx.items()]
best_exp_cashes_df = pd.concat(best_exp_cashes).reset_index(level="exp", drop=True).reorder_levels(["algo","date","test"])
best_exp_cashes_df = best_exp_cashes_df.groupby(level=["algo","date"]).mean().unstack(level="algo")

# Plot the Cashes over time resampled to weekly mean value
best_exp_cashes_df[drl_algos].resample("M").mean().plot(ylim=[100,400], title="Portfolio Cash (monthly mean", xlabel="Date", ylabel="Total Cash", color=drl_colors).figure.savefig("img/drl_cash.pdf")
# %%
# And for better comparison show the totalValue over time resampled to weekly mean value
portfolios_df[drl_algos].resample("M").mean().plot(title="Total portfolio value (monthly mean)", ylabel="Total Value", xlabel="Date", color=drl_colors).figure.savefig("img/drl_total_mean_monthly.pdf")
# %%
# What stocks did DDPG hold during that period?

# Get numShares in a multilevel DF for DDPG and PPO
best_exp_numshares = [tests_df["numShares"].loc[(algo_name, exp_idx, slice(None), slice(None))] for algo_name, exp_idx in best_exp_idx.items()]
best_exp_numshares_df = pd.concat(best_exp_numshares).reset_index(level="exp", drop=True).reorder_levels(["algo","date","test"]).to_frame()
# explode list fields to columns
best_exp_numshares_df = pd.DataFrame(data=best_exp_numshares_df["numShares"].values.tolist(), columns=stockprices_ser.index.get_level_values(level="symbol").unique().tolist(),index=best_exp_numshares_df.index)
# swap axis of test (index) and algo (column)
best_exp_numshares_df = best_exp_numshares_df.unstack(level="test").stack(level=None)
# reorder index levels
best_exp_numshares_df.index.names = ["algo", "date", "symbol"]
# get mean over tests, round the mean to full numbers and convert it to int.
best_exp_numshares_df = best_exp_numshares_df.mean(axis=1).round().astype("int").unstack(level="algo")
# copy to new df, because we want to have sharesvalues additionally
best_exp_sharesvalues_df = best_exp_numshares_df.copy()

# Get closing price and caclulate the stock value in portfolio
stockprices_ser.name = "close"
best_exp_sharesvalues_df = best_exp_sharesvalues_df.join(stockprices_ser)
for algo in drl_algos+basic_algos:
    best_exp_sharesvalues_df[algo] = best_exp_sharesvalues_df[algo] * best_exp_sharesvalues_df["close"]
best_exp_sharesvalues_df = best_exp_sharesvalues_df.drop(columns="close")
# %%
# Bar chart race for seperate symbol values
bcr.bar_chart_race(df=best_exp_sharesvalues_df["DDPG"].unstack("symbol").resample("2W").mean(), dpi=330,
                filename="vids/best_exp_sharesvalues_race_DDPG.mp4", orientation="v", interpolate_period=True,
                fixed_order=True, period_length=1000, filter_column_colors=True,
                fixed_max=True, steps_per_period=7, n_bars=10, cmap=[drl_colors[0]],
                title="Seperate Stock Values of DDPG over Time")
# %%
bcr.bar_chart_race(df=best_exp_numshares_df["DDPG"].unstack("symbol").resample("2W").mean(), dpi=330,
                filename="vids/best_exp_numshares_race_PPO.mp4", orientation="v", interpolate_period=True,
                fixed_order=True, period_length=1000, filter_column_colors=True,
                fixed_max=True, steps_per_period=7, n_bars=10, cmap=[drl_colors[0]],
                title="Seperate Stock numbers of DDPG over Time")

# %%
# The dominant stocks...
# Show the mean of each sharesvalue and plot the 10 largest.
best_exp_sharesvalues_df["DDPG"].unstack("symbol").mean().nlargest(10).plot(title="Mean sharevalue in DDPG Portfolio", ylabel="Value", xlabel="Symbol", kind="bar",color=drl_colors[0]).figure.savefig("img/ddpg_shares_values_mean.pdf")

# %%
#### How could PPO make such a rise up in the end?
# There must be one or more stocks that drived this rise.

bcr.bar_chart_race(df=best_exp_sharesvalues_df["PPO"].unstack("symbol").resample("2W").mean(), dpi=330,
                filename="vids/best_exp_sharesvalues_race_PPO.mp4", orientation="v", interpolate_period=True,
                fixed_order=True, period_length=1000, filter_column_colors=True,
                fixed_max=True, steps_per_period=7, n_bars=10, cmap=[drl_colors[1]],
                 title="Seperate Stock Values of PPO over Time")
# %%
bcr.bar_chart_race(df=best_exp_numshares_df["PPO"].unstack("symbol").resample("2W").mean(), dpi=330,
                filename="vids/best_exp_numshares_race_PPO.mp4", orientation="v", interpolate_period=True,
                fixed_order=True, period_length=1000, filter_column_colors=True,
                fixed_max=True, steps_per_period=7, n_bars=10, cmap=[drl_colors[1]],
                title="Seperate Stock numbers of PPO over Time")

# %%
# The dominant stocks...
# Show the mean of each sharesvalue and plot the 10 largest.
best_exp_sharesvalues_df["PPO"].unstack("symbol").mean().nlargest(10).plot(title="Mean sharevalue in PPO Portfolio", ylabel="Value", xlabel="Symbol", kind="bar",color=[drl_colors[1]]).figure.savefig("img/ppo_shares_values_mean.pdf")

# %%
#### What happend with A2C performance?
bcr.bar_chart_race(df=best_exp_sharesvalues_df["A2C"].unstack("symbol").resample("2W").mean(), dpi=330,
                filename="vids/best_exp_sharesvalues_race_A2C.mp4", orientation="v", interpolate_period=True,
                fixed_order=True, period_length=1000, filter_column_colors=True,
                fixed_max=True, steps_per_period=7, n_bars=10, cmap=[drl_colors[2]],
                title="Seperate Stock Values of A2C over Time")
# %%
bcr.bar_chart_race(df=best_exp_numshares_df["A2C"].unstack("symbol").resample("2W").mean(), dpi=330,
                filename="vids/best_exp_numshares_race_A2C.mp4", orientation="v", interpolate_period=True,
                fixed_order=True, period_length=1000, filter_column_colors=True,
                fixed_max=True, steps_per_period=7, n_bars=10, cmap=[drl_colors[2]],
                title="Seperate Stock Values of A2C over Time")

# %%
# The dominant stocks...
# Show the mean of each sharesvalue and plot the 10 largest.
best_exp_sharesvalues_df["A2C"].unstack("symbol").mean().nlargest(10).plot(title="Mean sharevalue in A2C Portfolio", ylabel="Value", xlabel="Symbol" ,kind="bar",color=drl_colors[2]).figure.savefig("img/a2c_shares_values_mean.pdf")

# %%
# Show the mean percentage of portfolio structur
with open(Path.cwd().parent / "etl" / "stocks.txt") as file:
    symb_exchange = file.read().splitlines()

stocks = pd.DataFrame([item.split(".") for item in symb_exchange], columns=["symbol","exchange"]).set_index("symbol")["exchange"].apply(lambda x: "EU" if x != "US" else x)

temp = [pd.Series(data=best_exp_sharesvalues_df[algo].unstack("symbol").mean().nlargest(10), name=algo) for algo in drl_algos]
exchange_values_mean_df = (pd.DataFrame(data=temp).T.sort_index().rename_axis(index="symbol").fillna(best_exp_sharesvalues_df.unstack("symbol").mean().unstack().T) \
    .join(stocks).set_index("exchange").sort_index().groupby(by="exchange").sum().T)
exchange_values_mean_df["Total"] = portfolios_df[drl_algos].mean()
exchange_values_mean_df["Cash"] = exchange_values_mean_df.apply(lambda x: x["Total"] - (x["EU"]+x["US"]), axis=1)
exchange_values_mean_df = exchange_values_mean_df[["EU","US","Cash","Total"]]
for column in exchange_values_mean_df.columns:
    exchange_values_mean_df[column] = exchange_values_mean_df.apply(lambda x: x[column]/x["Total"], axis=1)
exchange_values_mean_df = exchange_values_mean_df.drop(columns="Total")

exchange_values_mean_df.sort_index().plot(title="Mean Portfolio Structure by DRL Algorithm", ylabel="Percentage" ,kind="bar",stacked=True, legend="reverse",color=colors[::-1]).figure.savefig("img/drl_portfolio_structure_mean.pdf")

# %%
exchange_values_df = best_exp_sharesvalues_df.stack().copy()
exchange_values_df.index.names = ["date","symbol","algo"]
exchange_values_df.name = "value"

temp_cashes = best_exp_cashes_df.stack().copy()
temp_cashes.name = ("value","Cash")

exchange_values_df = exchange_values_df.reset_index(["date","algo"]).join(stocks).reset_index().set_index(["date","algo","exchange"]).drop(columns=["symbol"]).sort_index().groupby(["algo","date","exchange"]).sum().unstack("exchange").join(temp_cashes)

exchange_values_df[("value","Total")] = exchange_values_df[("value","US")] + exchange_values_df[("value","EU")] + exchange_values_df[("value","Cash")]
for column in exchange_values_df.columns:
    exchange_values_df[column] = exchange_values_df.apply(lambda x: x[column]/x[("value","Total")], axis=1)

exchange_values_df = exchange_values_df.drop(columns=("value","Total"))
# %%
exchange_values_df.groupby("algo").plot(kind="box")
# This plots on the one hand opens up some questions
# and on the other hand shows aspects very clearly

# Clear Aspects
# DDPG and PPO found out which stocks to trade for a positive effect to the portfolio
# A2C didn't found this out that concisely

# New Questions:
# DDPG and PPO had more portion of EU Stocks than US ones
# But the Euro Stoxx 50 didn't went that good in the second half of testing period
# One assumption could be, that in DJIA had been a few stocks that propelled the index
# On the other side in EStoxx50 it could be that there heavy weighted titles that throttled the index
# The question is (if the assumptions are true): Did DDPG and PPO found that out and picked the good ones / threw the bad ones away?

# %%
# A completely different Question is about the produced costs
# What algorithm traded the most cost effective?
# ... produced the most costs?
# ... made the minimum trades?

# Firstly checkout the Trades
best_exp_trades_df = best_exp_numshares_df.groupby(by="symbol").diff().fillna(0)

best_exp_trades_df.abs().groupby(by="date").sum().cumsum()[show_portfolios[:-1]].resample("W").mean().plot(color=colors, xlabel="Date", ylabel="Trades", title="Cummulative Number of Trades").figure.savefig("img/number_of_trades.pdf")
# %%
# Secondly the costs
best_exp_costs_df = best_exp_trades_df.copy()
stockopen_ser = stocksdata_df["open"].copy()
stockopen_ser.name = "open"
best_exp_costs_df = best_exp_costs_df.join(stockopen_ser)

for algo in drl_algos+basic_algos:
    best_exp_costs_df[algo] = best_exp_costs_df[algo].abs() * best_exp_costs_df["open"] * best_exp_args_df.loc[(algo,slice(None)),"fee"].values[0]
best_exp_costs_df = best_exp_costs_df.drop(columns="open")

best_exp_costs_df.groupby(by="date").sum().cumsum()[show_portfolios[:-1]].resample("W").mean().plot(color=colors, xlabel="Date", ylabel="Costs", title="Cummulative Trading Costs").figure.savefig("img/trading_costs.pdf")
# %%
