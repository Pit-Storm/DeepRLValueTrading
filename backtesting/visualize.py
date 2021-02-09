# %%
import sys
sys.path.append("..")
from data import handling as dth
import pandas as pd
from pathlib import Path
%matplotlib inline
import quantstats as qs
# extend pandas functionality with metrics, etc.
qs.extend_pandas()
# %%
algo_name = "BuyHold" # What algo to evaluate. must be the same as in run_DRL.py
rewards_ser = None # timeseries with dates and rewards
portfolio_df = None # index (Dates), [symbols, ...] (float: net amount holding), cash (float)
actions_df = None # index (date), amount (int: action), price (float: open), symbol (string)
indices_df = None # Timeseries with dates and daily returns
sum_mode = "comp" # How will the growth rate be calculated? Compound or Cummulative

algo_dir = Path.cwd().parent / "results" / algo_name
run_dirs = [dr for dr in algo_dir.iterdir() if dr.is_dir()]
ep_path = run_dirs[-1] / "env_info" / "test" / "episode_0001.json"

env_info_df = (pd.read_json(ep_path,  convert_dates=["dates"])
                .set_index("dates"))

rewards_ser = env_info_df["rewards"].copy().rename(index="dly_ret").rename_axis("Date")
# %%
indices_dir = Path.cwd().parent / "data" / "indices_performance.csv"
indices_df = dth.load_data(indices_dir)
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
# %%
qs.plots.snapshot(rewards_ser, title=algo_name+" Performance", mode=sum_mode)
# %%
qs.plots.snapshot(stoxx50e_ser, title="EuroStoxx50 Performance", mode=sum_mode)
# %%
qs.plots.snapshot(dji_ser,title="Dow Jones Industrial Average Performance", mode=sum_mode)
