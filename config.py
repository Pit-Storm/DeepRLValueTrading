from datetime import datetime
from pathlib import Path, PurePath
import configargparse as argparse
import json

basic_algos = ["BuyHold", "Random"]
drl_algos = ["A2C", "PPO", "DDPG"]
algos = basic_algos + drl_algos
policies = ["MlpLstmPolicy", "MlpPolicy"]

parser = argparse.ArgumentParser(prog="DeepRLValueTrading",
            description="Train and Run different Deep RL Algos for trading. "+ \
                 "Some algorithms are there for backtesting the DRL ones.")
parser.add_argument("--algo", action="store", required=True, type=str, choices=algos)
parser.add_argument("--cash", action="store", default=1000000, type=int)
parser.add_argument("--episodic", action="store_true", default=False)
parser.add_argument("--fee", action="store", default=0.001, type=float)
parser.add_argument("--learn_steps", action="store", default=15000, type=int)
parser.add_argument("--load", action="store", default=None, type=str, help="Path to load a model and evaluate it.")
parser.add_argument("--policy", action="store", default="MlpLstmPolicy", type=str, choices=policies)
parser.add_argument("--result_dir", action="store", default="results", type=str)
parser.add_argument("--scaling", action="store", default=100, type=int)
parser.add_argument("--test_eps", action="store", default=100, type=int)
parser.add_argument("--trainsampling", action="store_true", default=False)
parser.add_argument("--val_eps", action="store", default=10, type=int)
parser.add_argument("--val_freq", action="store", default=1000, type=int)
parser.add_argument("--yearrange", action="store", default=4, type=int)
args = parser.parse_args()

# How much cash does the agent have from beginning?
INIT_CASH = args.cash
# How much percent of the trade (num times price) will a trade cost?
TRADE_FEE_PRCT = args.fee
# Scale the generated actions by this value
# to get the actual number of trades per share per step
# This is Equal to the maximum number of buy/sell actions per share and step
ACTION_SCALING = args.scaling

### PATHS
BASE_PATH = Path.cwd() / args.result_dir

### NAMES
MODEL_NAME = args.algo
POLICY = args.policy

### VARS
seed = 42
train_envs = 1
val_envs = 1
# Tensorboard Logs path slice
TB_LOGS_PATH = "tb_logs"
# Best Model save path slice
BEST_MODELS_PATH = "val_info"
# Env info save path
ENV_INFO_PATH = "env_info"


# set the vars out of args
yearrange = args.yearrange
episodic = args.episodic
trainsampling = args.trainsampling
val_freq = args.val_freq
val_eps = args.val_eps
test_eps = args.test_eps
learn_steps = args.learn_steps
timestr = datetime.now().strftime('%Y-%m-%d_%H-%M')

# Create paths and make checks
base_path = BASE_PATH / MODEL_NAME / timestr
assert isinstance(base_path, PurePath), "save_path is no pathlib.Path object."
base_path.mkdir(parents=True, exist_ok=True)
env_path = base_path / ENV_INFO_PATH
tb_path = base_path / TB_LOGS_PATH
val_path = base_path / BEST_MODELS_PATH
data_path = Path.cwd().joinpath("data","stocksdata_all.csv")

with open(base_path.joinpath('run_args.json'), 'w') as fp:
    json.dump(vars(args), fp, indent=4)