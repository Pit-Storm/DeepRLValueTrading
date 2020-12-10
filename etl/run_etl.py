from os import system

# load the components of indices
# Only used if you want to update them
# IMPORTANT: Be aware that there had been some manual modifications to stocks.txt
# system("python components.py")

# Load all used data for the symbols
system("python trading_data.py")
system("python fundamentaldata.py")
system("python forex_data.py")
# stich load data together
system("python get_all_attributes.py")

# Load Indices data for backtesting
system("python indices_performance.py")