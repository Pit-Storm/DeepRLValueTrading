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
# %%
# Generate Multiindex DF for all test results over all experiments over all algos
results_dir = Path.cwd().parent / "results"
tests = []

algo_dirs = [algo_dir for algo_dir in results_dir.iterdir() if algo_dir.is_dir()]
for algo_dir in algo_dirs:
    exp_dirs = [exp_dir for exp_dir in algo_dir.iterdir() if exp_dir.is_dir()]
    for exp_idx, exp_dir in enumerate(exp_dirs):
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

# Get Sharpe ratio for all tests
tests_sharpe = tests_df.groupby(level=["algo", "exp", "test"]).apply(qs.stats.sharpe).rename(columns={"totalValues": "sharpe"})
# Get mean sharpe over the tests for each run
tests_sharpe_mean = tests_sharpe.groupby(level=["algo", "exp"]).mean()
# save best run idx to a dict with algo names as keys
best_exp_idx = dict([item for sublist in tests_sharpe_mean.groupby(level=["algo"]).idxmax().values.tolist() for item in sublist])

# Inside the best experiment, calculate the mean totalValue by date over all tests.
# for that we have to slice out the best exp out of our overall df
best_exp = []
for algo_name, exp_idx in best_exp_idx.items():
    best_exp.append(tests_df.loc[(algo_name, exp_idx, slice(None), slice(None)), slice(None)])

best_exps_df = pd.concat(best_exp).reset_index(level="exp", drop=True).reorder_levels(["algo","date","test"])

###
# Now we getting our data to make graphs and other metrics.
best_exp_mean_df = best_exps_df.groupby(level=["algo","date"]).mean().unstack(level="algo")
# Get rid of multilevel column names.
best_exp_mean_df.columns = best_exp_mean_df.columns.droplevel(None)
# %%
### Create indices series
indices_fp = Path.cwd().parent / "data" / "indices_performance.csv"
indices_df = dth.load_data(indices_fp)
*_, indices_df = dth.train_val_test_split(indices_df)

stoxx50e_ser = indices_df[indices_df.index.get_level_values("symbol") == "stoxx50e"]["adjusted_close"]
stoxx50e_ser.index = stoxx50e_ser.index.droplevel(1)
stoxx50e_ser = stoxx50e_ser.pct_change()
stoxx50e_ser = stoxx50e_ser.fillna(0)
stoxx50e_ser = stoxx50e_ser.rename(index="dly_ret").rename_axis("Date")
stoxx50e_ser.name = "EuroStoxx50"

dji_ser =  indices_df[indices_df.index.get_level_values("symbol") == "dji"]["adjusted_close"]
dji_ser.index = dji_ser.index.droplevel(1)
dji_ser = dji_ser.pct_change()
dji_ser = dji_ser.fillna(0)
dji_ser = dji_ser.rename(index="dly_ret").rename_axis("Date")
dji_ser.name = "DJIA"

# TODO: create indices portfolio
    # percentage of invested capital as distribution of assets tradable for algo
    # This will be the real benchmark
    # 36/63 EU shares and 27/63 US shares
    # stoxx50e = 36 / 63 and dji = 1 - (36/63)
# %%
etf_df = pd.DataFrame(index=dates)
etf_df = etf_df.join([dji_ser, stoxx50e_ser]).fillna(0)

# %%
# TODO: Calculate the overall dly return of our etf portfolio
# calculate columns with fraction of pct_change of index
    # eg. djia_fract = dija * (1-(36/63))
    # and stoxx50_fract = stoxx50 * (36/63)
# calculate column with overall pct_change
    # eg. djia_fract + stoxx_fract
# It gets the weighted daily returns
# To calculate a portfolio value one has to do
# weighted_rets.add(1).mul(1e6)

# %%
# Bar chart race for Portfolio values
bcr.bar_chart_race(df=best_exp_mean_df, dpi=330,
                filename="best_exp_mean_totalValues_race.mp4", orientation="v",
                fixed_order=False, period_length=200,
                fixed_max=True, steps_per_period=1,
                title="Portfolio Values over Time")

# %%
# Matrix of Mean Sharpes (Experiment times Algorithms)
mean_sharpes_df = tests_sharpe_mean.unstack(level=["algo"])
mean_sharpes_df.columns = mean_sharpes_df.columns.droplevel(None)
# Best experiment sharpe is highlighted
mean_sharpes_df.style.highlight_max(axis=0)
# %%
# Table for evaluation of best experiments for algos and Bench with:
    # Comp. overall Return
    # Comp. annual growth rate
    # Sharpe ratio
metrics_df = pd.DataFrame(index=["return", "CAGR", "sharpe"])

for column in best_exp_mean_df.columns:
    sharpe = qs.stats.sharpe(qs.utils.to_returns(best_exp_mean_df[column]))
    ret = qs.stats.comp(qs.utils.to_returns(best_exp_mean_df[column]))
    cagr = qs.stats.cagr(qs.utils.to_returns(best_exp_mean_df[column]))
    metrics_df[column] = [ret, cagr, sharpe]

# Higlight best algorithm
metrics_df.style.highlight_max(axis=1)
# %%
# TODO: Investigations
    # 



# %%
