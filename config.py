from datetime import datetime
from os import path

# How much cash does the agent have from beginning?
INIT_CASH = 1000000
# How much percent of the trade (num times price) will a trade cost?
TRADE_FEE_PRCT = 0.001
# Scale the generated actions by this value
# to get the actual number of trades per share per step
ACTION_SCALING = 100

###
# Model PAths
MODEL_BASE_PATH = "./models/"
# A2C Best Model path
A2C_MODEL_PATH = path.join(MODEL_BASE_PATH, "a2c_best/")

# Tensorboard PAths
TB_BASE_PATH = "./tb-logs/"
# A2C Logs path
A2C_TB_PATH = path.join(TB_BASE_PATH, "a2c/")