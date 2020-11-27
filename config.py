from datetime import datetime
from pathlib import Path

# How much cash does the agent have from beginning?
INIT_CASH = 1000000
# How much percent of the trade (num times price) will a trade cost?
TRADE_FEE_PRCT = 0.001
# Scale the generated actions by this value
# to get the actual number of trades per share per step
ACTION_SCALING = 100

### PATHS
BASE_PATH = Path.cwd() / "results"
# Tensorboard Logs path slice
TB_LOGS_PATH = "tb_logs"
# Best Model save path slice
BEST_MODELS_PATH = "best_models"
# Env info save path
ENV_INFO_PATH = "env_info"

### NAMES
MODEL_NAME = "RANDOM"