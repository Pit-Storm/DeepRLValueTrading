# %%
import sys
sys.path.append("..")
from data import handling as dth
import pandas as pd
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
stocksdata_df = dth.load_data(stocksdata_fp)
*_, stocksdata_df = dth.train_val_test_split(stocksdata_df)
dates = stocksdata_df.index.get_level_values(level="date").unique().tolist()
###
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
            total_values = pd.read_json(test_file)["totalValues"]
            data = {
                "algo": algo_dir.name,
                "exp": exp_idx,
                "test": test_idx,
                "date": dates,
                "totalValues": total_values
                }
            tests.append(pd.DataFrame(data=data))

# concatenate all tests to one df
tests_df = pd.concat(tests).set_index(["algo", "exp", "test", "date"])
# and all exp_args to one df
exp_args_df = pd.DataFrame(exp_args)
exp_args_df["exp_idx"] = exp_args_df.groupby(by="algo").cumcount()
exp_args_df = exp_args_df.set_index(["algo","exp_idx"])

# Get Sharpe ratio for all tests
tests_sharpe = tests_df.groupby(level=["algo", "exp", "test"]).apply(qs.stats.sharpe).rename(columns={"totalValues": "sharpe"})
# Get mean sharpe over the tests for each run
tests_sharpe_mean = tests_sharpe.groupby(level=["algo", "exp"]).mean()
# save best run idx to a dict with algo names as keys
best_exp_idx = pd.Series(dict([item for sublist in tests_sharpe_mean.groupby(level=["algo"]).idxmax().values.tolist() for item in sublist]))

# Inside the best experiment, calculate the mean totalValue by date over all tests.
# for that we have to slice out the best exp out of our overall df
best_exp = [tests_df.loc[(algo_name, exp_idx, slice(None), slice(None)), slice(None)] for algo_name, exp_idx in best_exp_idx.items()]
best_exps_df = pd.concat(best_exp).reset_index(level="exp", drop=True).reorder_levels(["algo","date","test"])

###
# Now we getting our data to make graphs and other metrics.
best_exp_mean_df = best_exps_df.groupby(level=["algo","date"]).mean().unstack(level="algo")
# Get rid of multilevel column names.
best_exp_mean_df.columns = best_exp_mean_df.columns.droplevel(None)

###
# Creating the best exps_args_df to show it later
best_exp_args = [exp_args_df.loc[(algo_name, exp_idx), slice(None)] for algo_name, exp_idx in best_exp_idx.items()]
best_exps_args_df = pd.DataFrame(best_exp_args).reset_index()
best_exps_args_df[['algo','exp_idx']] = pd.DataFrame(best_exps_args_df["index"].tolist(), index= best_exps_args_df.index)
best_exps_args_df = best_exps_args_df.drop(columns="index").set_index(["algo","exp_idx"])

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
# Create a DF with all portfolio values
portfolios_df = best_exp_mean_df.join(etf_ser)
# %%
# Bar chart race for Portfolio values
# bcr.bar_chart_race(df=portfolios_df, dpi=330,
#                 filename="best_exp_mean_totalValues_race.mp4", orientation="v",
#                 fixed_order=False, period_length=200,
#                 fixed_max=True, steps_per_period=1,
#                 title="Portfolio Values over Time")

# %%
drl_algos = ["A2C", "DDPG", "PPO"]

# Matrix of Mean Sharpes (Experiment times Algorithms)
mean_sharpes_df = tests_sharpe_mean.unstack(level=["algo"])
mean_sharpes_df.columns = mean_sharpes_df.columns.droplevel(None)
# Best experiment sharpe is highlighted
mean_sharpes_df[drl_algos].style.highlight_max(axis=0)

# %%
# Showing the variant args of the best experiments
variant_args = ["cagr", "episodic","num_stacks", "trainsampling"]
best_exps_args_df[variant_args].loc[drl_algos]
# %%
# Table for evaluation of best experiments for algos and Bench with:
    # Comp. overall Return
    # Comp. annual growth rate
    # Sharpe ratio
    # Volatility (Standard deviation p.a.)
metrics_df = pd.DataFrame(index=["Return", "Volatility p.a.", "CAGR", "Exp. Return p.a.", "Sharpe ratio"])

interval = "M"
for column in portfolios_df.columns:
    sharpe = qs.stats.sharpe(qs.utils.to_returns(portfolios_df[column].resample(interval).last()))
    ret = qs.stats.comp(qs.utils.to_returns(portfolios_df[column].resample(interval).last()))
    cagr = qs.stats.cagr(qs.utils.to_returns(portfolios_df[column].resample(interval).last()))
    vol = qs.stats.volatility(qs.utils.to_returns(portfolios_df[column].resample(interval).last()), periods=12)
    exp_ret = qs.stats.expected_return(qs.utils.to_returns(portfolios_df[column].resample(interval).last()), aggregate="M")
    metrics_df[column] = [ret, vol/100, cagr, exp_ret, sharpe]

# Higlight best algorithm
metrics_df.style.highlight_max(axis=1)
# %%
# Plot a linechart for all portfolios resampled to monthly last value.
portfolios_df.resample("M").last().plot()

# %%
# TODO: Investigations
#### Why DDPG wasn't effected by the drop in december 2018?
# The news can tell us, that in december 2018 the US stock market had a historical drop
# Let's chack this by plotting the reshaped performance of DJIA and EuroStoxx50
etf_df[["dji", "stoxx50e"]].add(1).cumprod().resample("M").last().plot()
# We can see a drop in DJIA, but also in EuroStoxx50...
# What stocks did DDPG hold during that period?
# Did it hold more or less cash during the period?

# %%
#### How could PPO make such a rise up in the end?
# There must be one or more stocks that drived this rise.
# For that one hase to caclulate the percentage of stockvalue on totalValue for each stock.
# 

# %%
#### What happend with A2C performance?
# Theory: It is not fully trained or didn't converge.
# Reason could be the relative bad sample efficiency of A2C
# Another theory could be, that A2C learned to reduce variance/Volatility.
# If you reduce volatility in your portfolio, you avoid losses.
# But in addition you avoid wins. The reason fort hat is, because standard deviation swings to both sides. Up and down.

# %%
