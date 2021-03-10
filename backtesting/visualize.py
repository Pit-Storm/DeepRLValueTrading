# %%
import sys
sys.path.append("..")
from data import handling as dth
import pandas as pd
from pathlib import Path
%matplotlib inline
import quantstats as qs
import bar_chart_race as bcr
# extend pandas functionality with metrics, etc.
qs.extend_pandas()
# %%
# TODO: Do this for every algorithm in results dir programatically
algo_name = "DDPG" # What algo to evaluate. must be the same as specified in config.py
algo_dir = Path.cwd().parent / "results" / algo_name.upper()
# get all rundirs with all test jsons inside
run_dirs_files = [[fp for fp in run_dir.joinpath("env_info").rglob("test/*.json") if fp.is_file()] for run_dir in algo_dir.iterdir() if run_dir.is_dir()]

# %%
# TODO: Create dataStructure that can hold all algos programatically
# get all Tests for every run
runs = []
for run_dir in run_dirs_files:
    temp = [pd.read_json(file, convert_dates=["dates"]).set_index("dates") for file in run_dir]
    temp_df = pd.concat([df["totalValues"] for df in temp], axis=1).sort_index()
    temp_df.columns = [f"test_{i}" for i in range(len(run_dir))]
    temp_df["tests_mean"] = temp_df.mean(axis="columns")
    temp_df["tests_std"] = temp_df.std(axis="columns")
    runs.append({"values_df": temp_df.copy()})

# Get best run by Sharpe
runs_mean_sharpe = pd.DataFrame({"sharpe": [qs.stats.sharpe(qs.utils.to_returns(runs[i]["values_df"]["tests_mean"])) for i in range(len(runs))]})
best_run_idx = runs_mean_sharpe.idxmax()[0]

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

dji_ser =  indices_df[indices_df.index.get_level_values("symbol") == "dji"]["adjusted_close"]
dji_ser.index = dji_ser.index.droplevel(1)
dji_ser = dji_ser.pct_change()
dji_ser = dji_ser.fillna(0)
dji_ser = dji_ser.rename(index="dly_ret").rename_axis("Date")

# TODO: create indices portfolio
    # percentage of invested capital as distribution of assets tradable for algo
    # This will be the real benchmark

# %%
# TODO: Animate totalValues of every algo in one video.
# bcr.bar_chart_race(df=numShares_df, 
#                 filename="numShares_race_"+ algo_name.lower() +".mp4", orientation="v",
#                 n_bars=21, fixed_order=True, period_length=200,
#                 fixed_max=True, steps_per_period=1)

# %%

