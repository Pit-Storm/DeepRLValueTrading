from datetime import datetime
from pathlib import Path, PurePath
import argparse
import json
import sys

# Vars for use in argparse and program flow
basic_algos = ["BuyHold", "Random"]
drl_algos = ["A2C", "PPO", "DDPG"]
algos = basic_algos + drl_algos
policies = ["MlpLstmPolicy", "MlpPolicy"]

# argparse arguments
parser = argparse.ArgumentParser(description="Train and Evaluate different Deep RL Algos for trading. Some algorithms are there for backtesting the DRL ones.")
parser.add_argument("--algo", action="store", required=True, type=str, choices=algos, help="Choose the algorithm to train and evaluate. Is required.")
parser.add_argument("--cash", action="store", default=1000000, type=int, help="Initial cash the algo can spent.")
parser.add_argument("--deterministic", action="store_true", default=False, help="If you set this, val_eps and test_eps will be 1")
parser.add_argument("--episodic", action="store_true", default=False, help="Set it to give a cumulated reward on the end of period. If unset step per step reward.")
parser.add_argument("--fee", action="store", default=0.001, type=float, help="Percentage of costs per trade.")
parser.add_argument("--learn_steps", action="store", default=15000, type=int, help="Number of timesteps the Agent will learn. Only for DRL algos.")
parser.add_argument("--policy", action="store", default="MlpLstmPolicy", type=str, choices=policies, help="DDPG always uses non recurrent.")
parser.add_argument("--result_dir", action="store", default="results", type=str, help="Base folder to store results in. Default: %(default)s")
parser.add_argument("--scaling", action="store", default=100, type=int, help="Max possible trades per share per step.")
parser.add_argument("--test_eps", action="store", default=100, type=int, help="Takes effect if --deterministic is unset.")
parser.add_argument("--trainsampling", action="store_true", default=False, help="Sample --yearrange timeperiod out of training data for every episode.")
parser.add_argument("--val_eps", action="store", default=10, type=int, help="Takes effect if --deterministic is unset.")
parser.add_argument("--val_freq", action="store", default=1000, type=int, help="Every VAL_FREQth timestep the agent will be evaluated. Only for DRL.")
parser.add_argument("--yearrange", action="store", default=4, type=int, help="The yearrange of train and test env data. In combination with --trainsampling you additionally control the range of the sampled period.")
args = parser.parse_args()

# How much cash does the agent have from beginning?
INIT_CASH = args.cash
# How much percent of the trade (num times price) will a trade cost?
TRADE_FEE_PRCT = args.fee
# Scale the generated actions by this value
# to get the actual number of trades per share per step
# This is Equal to the maximum number of buy/sell actions per share and step
ACTION_SCALING = args.scaling

BASE_PATH = Path.cwd() / args.result_dir
MODEL_NAME = args.algo
args.policy = "MlpPolicy" if MODEL_NAME == "DDPG" else args.policy
POLICY = args.policy

### VARS
seed = 42
num_envs = 1
# Tensorboard Logs path slice
TB_LOGS_PATH = "tb_logs"
# Best Model save path slice
BEST_MODELS_PATH = "val_info"
# Env info save path
ENV_INFO_PATH = "env_info"


# set the vars out of args
episodic = args.episodic
deterministic = args.deterministic
learn_steps = args.learn_steps
test_eps = 1 if deterministic else args.test_eps
timestr = datetime.now().strftime('%Y-%m-%d_%H-%M')
trainsampling = args.trainsampling
val_eps = 1 if deterministic else args.val_eps
val_freq = args.val_freq
yearrange = args.yearrange

# Create paths
base_path = BASE_PATH / MODEL_NAME / timestr
base_path.mkdir(parents=True, exist_ok=True)
env_path = base_path / ENV_INFO_PATH
tb_path = base_path / TB_LOGS_PATH
val_path = base_path / BEST_MODELS_PATH
data_path = Path.cwd().joinpath("data","stocksdata_all.csv")

with open(base_path.joinpath("run_args.json"), "w") as fp:
    json.dump(vars(args), fp, indent=4, sort_keys=True)

with open(base_path.joinpath("command.txt"), "w") as fp:
    fp.write(" ".join(sys.argv))