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
######
# SETUP
######
render_vids = False

######
# PREPROCESSING
######

# Load stocksdata
stocksdata_fp = Path.cwd().parent / "data" / "stocksdata_all.csv"
stocksdata_all_df = dth.load_data(stocksdata_fp)
*_, stocksdata_df = dth.train_val_test_split(stocksdata_all_df)

# Get tested dates out of stocksdata
dates = stocksdata_df.index.get_level_values(level="date").unique().tolist()
# Get stock prices close and open
stockprices_close_ser = stocksdata_df["close"].copy()
stockprices_close_ser.name = "close"
stockprices_open_ser = stocksdata_df["open"].copy()
stockprices_open_ser.name = "open"


# Load indices data
indices_fp = Path.cwd().parent / "data" / "indices_performance.csv"
indices_df = dth.load_data(indices_fp)
*_, indices_df = dth.train_val_test_split(indices_df)

#######
# READ TEST RESULTS
######

# Generate Multiindex DF for all test results over all experiments over all algos
results_dir = Path.cwd().parent / "results"
tests = []
exp_args = []

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

###
# Get the best Experiment by Sharpe Ratio

# Get Sharpe ratio for all tests
tests_sharpe = tests_df["totalValues"].groupby(level=["algo", "exp", "test"]).apply(qs.stats.sharpe).rename("sharpe")
# Get mean sharpe over the tests for each run
tests_sharpe_mean = tests_sharpe.groupby(level=["algo", "exp"]).mean()
# save best exp idx to a pandas series
best_exp_idx = pd.Series(dict(tests_sharpe_mean.groupby(level=["algo"]).idxmax().values.tolist())).rename("best_exp_idx")

###
# Creating the best exps_args_df to show it later
temp = [exp_args_df.loc[(algo_name, exp_idx), slice(None)] for algo_name, exp_idx in best_exp_idx.items()]
best_exp_args_df = pd.DataFrame(temp).reset_index()
best_exp_args_df[['algo','exp_idx']] = pd.DataFrame(best_exp_args_df["index"].tolist(), index= best_exp_args_df.index)
best_exp_args_df = best_exp_args_df.drop(columns="index").set_index(["algo","exp_idx"])

######
# BUILD PORTFOLIO DF
######

# Inside the best experiment, calculate the mean totalValue by date over all tests.
temp = [tests_df["totalValues"].loc[(algo_name, exp_idx, slice(None), slice(None))] for algo_name, exp_idx in best_exp_idx.items()]
best_exp_totalValues_df = pd.concat(temp).reset_index(level="exp", drop=True).reorder_levels(["algo","date","test"])
# Now we getting our data to make graphs and other metrics.
best_exp_totalValues_df = best_exp_totalValues_df.groupby(level=["algo","date"]).mean().unstack(level="algo")

### Create indices series
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
etf_df["dji"] = etf_df["dji"]*(1-(36/63))
etf_df["stoxx50e"] = etf_df["stoxx50e"]*(36/63)
etf_df["portf_ret"] = etf_df["dji"] + etf_df["stoxx50e"]
# Calculate the portfolio value
etf_df["totalValue"] = etf_df["portf_ret"].add(1).cumprod().mul(1e6)

etf_ser = etf_df["totalValue"]
etf_ser.name = "ETF"

###
# Get all TotalValues in one DF
portfolios_df = best_exp_totalValues_df.join(etf_ser)

# %%
######
# SHOW RESULTS
######

# Setup the order
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

# Plot a linechart for all portfolios
portfolios_df[show_portfolios].plot(title="Total portfolio value", legend=True, xlabel="Date", ylabel="Total Value", color=colors).figure.savefig("img/all_line_totalValues.pdf")
# %%
# Bar chart race for Portfolio values
if render_vids:
    bcr.bar_chart_race(df=portfolios_df.resample("2W").mean(), dpi=330, cmap=[colors[1]]+[colors[0]]+colors[2:],
                    filename="vids/all_bcr_totalValues.mp4", orientation="v",
                    fixed_order=True, period_length=1000, interpolate_period=True,
                    fixed_max=True, steps_per_period=7,
                    title="Portfolio Values over Time")

# %%
######
# INVESTIGATIONS
######

# We can investigate the following underlying numbers:
    # Number and Value of Shares
    # Portfolio structure grouped by exchanges (US, EU) and Cash
    # Trading Costs

######
# NUMBER AND VALUE OF SHARES

# Get numShares in a multilevel DF
temp = [tests_df["numShares"].loc[(algo_name, exp_idx, slice(None), slice(None))] for algo_name, exp_idx in best_exp_idx.items()]
best_exp_numshares_df = pd.concat(temp).reset_index(level="exp", drop=True).reorder_levels(["algo","date","test"]).to_frame()
# explode list fields to columns
best_exp_numshares_df = pd.DataFrame(data=best_exp_numshares_df["numShares"].values.tolist(), columns=stockprices_close_ser.index.get_level_values(level="symbol").unique().tolist(),index=best_exp_numshares_df.index)
# swap axis of test (index) and algo (column)
best_exp_numshares_df = best_exp_numshares_df.unstack(level="test").stack(level=None)
# reorder index levels
best_exp_numshares_df.index.names = ["algo", "date", "symbol"]
# get mean over tests, round the mean to full numbers and convert it to int.
best_exp_numshares_df = best_exp_numshares_df.mean(axis=1).round().astype("int").unstack(level="algo")
# copy to new df, because we want to have sharesvalues additionally
best_exp_sharesvalues_df = best_exp_numshares_df.copy()

# Get closing price and calculate the stock value in portfolio
best_exp_sharesvalues_df = best_exp_sharesvalues_df.join(stockprices_close_ser)
for algo in drl_algos+basic_algos:
    best_exp_sharesvalues_df[algo] = best_exp_sharesvalues_df[algo] * best_exp_sharesvalues_df["close"]
best_exp_sharesvalues_df = best_exp_sharesvalues_df.drop(columns="close")
# %%
for idx,algo in enumerate(drl_algos):
    if render_vids:
        # Bar chart race for seperate symbol values
        bcr.bar_chart_race(df=best_exp_sharesvalues_df[algo].unstack("symbol").resample("2W").mean(), dpi=330,
                        filename="vids/"+algo+"_bcr_sharesvalues.mp4", orientation="v", interpolate_period=True,
                        fixed_order=True, period_length=1000, filter_column_colors=True,
                        fixed_max=True, steps_per_period=7, n_bars=10, cmap=[drl_colors[idx]],
                        title="Seperate Stock Values of "+algo+" over Time")
        # Bar chart race for seperate symbol number of shares
        bcr.bar_chart_race(df=best_exp_numshares_df[algo].unstack("symbol").resample("2W").mean(), dpi=330,
                        filename="vids/"+algo+"_bcr_numshares.mp4", orientation="v", interpolate_period=True,
                        fixed_order=True, period_length=1000, filter_column_colors=True,
                        fixed_max=True, steps_per_period=7, n_bars=10, cmap=[drl_colors[idx]],
                        title="Seperate Stock numbers of "+algo+" over Time")


    # Show the mean of each sharesvalue and plot the 10 largest.
    best_exp_sharesvalues_df[algo].unstack("symbol").mean().nlargest(10).plot(title="Mean sharevalue in "+algo+" Portfolio", ylabel="Value", xlabel="Symbol", kind="bar",color=drl_colors[idx]).figure.savefig("img/"+algo+"_bar_top10_shares_values.pdf")
    # Show the mean of each numshares and plot the 10 largest.
    best_exp_numshares_df[algo].unstack("symbol").mean().nlargest(10).plot(title="Mean count of Shares in "+algo+" Portfolio", ylabel="Count", xlabel="Symbol", kind="bar",color=drl_colors[idx]).figure.savefig("img/"+algo+"_bar_top10_shares_counts.pdf")

# %%
######
# PORTFOLIO STRUCTURE

# cash 

# Get the Cashes of the Algorithms
temp = [tests_df["cashes"].loc[(algo_name, exp_idx, slice(None), slice(None))] for algo_name, exp_idx in best_exp_idx.items()]
best_exp_cashes_df = pd.concat(temp).reset_index(level="exp", drop=True).reorder_levels(["algo","date","test"])
best_exp_cashes_df = best_exp_cashes_df.groupby(level=["algo","date"]).mean().unstack(level="algo")

######
# Structure

with open(Path.cwd().parent / "etl" / "stocks.txt") as file:
    symb_exchange = file.read().splitlines()
stocks_df = pd.DataFrame([item.split(".") for item in symb_exchange], columns=["symbol","exchange"]).set_index("symbol")["exchange"].apply(lambda x: "EU" if x != "US" else x)

exchange_values_df = best_exp_sharesvalues_df.stack().copy()
exchange_values_df.index.names = ["date","symbol","algo"]
exchange_values_df.name = "value"

temp = best_exp_cashes_df.stack().copy()
temp.name = ("value","Cash")

exchange_values_df = exchange_values_df.reset_index(["date","algo"]).join(stocks_df).reset_index().set_index(["date","algo","exchange"]).drop(columns=["symbol"]).sort_index().groupby(["algo","date","exchange"]).sum().unstack("exchange").join(temp)
exchange_values_df.columns = exchange_values_df.columns.droplevel(None)

# Show absolute Structure of Portfolios
for algo in exchange_values_df.index.get_level_values(level="algo").unique().tolist():
    exchange_values_df.loc[(algo,slice(None)),slice(None)].plot(kind="area", title="Portfolio Structure of "+algo, ylabel="Value", legend="reverse",color=colors[::-1]).figure.savefig("img/"+algo+"_area_value_portfolio_structure.pdf")
# %%
# And for better comparison the percentage of total structure
exchange_pct_df = exchange_values_df.copy()
exchange_pct_df["Total"] = exchange_pct_df["US"] + exchange_pct_df["EU"] + exchange_pct_df["Cash"]
for column in exchange_pct_df.columns:
    exchange_pct_df[column] = exchange_pct_df.apply(lambda x: x[column]/x["Total"], axis=1)

exchange_pct_df = exchange_pct_df.drop(columns="Total")

# Show it...
for algo in exchange_pct_df.index.get_level_values(level="algo").unique().tolist():
    exchange_pct_df.loc[(algo,slice(None)),slice(None)].plot(kind="area", title="Portfolio Structure of "+algo, ylabel="Percentage", legend="reverse",color=colors[::-1]).figure.savefig("img/"+algo+"_area_pct_portfolio_structure.pdf")

# These plots on the one hand opens up some questions
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
######
# TRADING COSTS
######

# A completely different Question is about the produced costs
# What algorithm traded the most cost effective?

# Firstly we need the trades
best_exp_trades_df = best_exp_numshares_df.groupby(by="symbol").diff().fillna(0)

# and secondly we need the costs
best_exp_costs_df = best_exp_trades_df.copy()
best_exp_costs_df = best_exp_costs_df.join(stockprices_open_ser)

for algo in drl_algos+basic_algos:
    best_exp_costs_df[algo] = best_exp_costs_df[algo].abs() * best_exp_costs_df["open"] * best_exp_args_df.loc[(algo,slice(None)),"fee"].values[0]
best_exp_costs_df = best_exp_costs_df.drop(columns="open")

# Show the Marginal costs per trade over time
# Marginal costs determine how effective the trades has been made in conjuntion to costs
(best_exp_costs_df.groupby(by="date").sum().cumsum() / best_exp_trades_df.abs().groupby(by="date").sum().cumsum())[show_portfolios[:-1]].plot(ylabel="Cost/Trade", title="Marginal Cost per Trade over Time", color=colors, xlabel="Date").figure.savefig("img/all_line_marginal_trading_costs.pdf")
# %%
