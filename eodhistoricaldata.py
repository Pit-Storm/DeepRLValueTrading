###
# Imports
###

import os
import requests
import pandas as pd
from stockstats import StockDataFrame

###
# Constants
###

# API_KEY
API_KEY = None
def check_API_KEY():
    global API_KEY
    if os.environ.get("API_KEY") == None:
        raise NameError("Set env var API_KEY befor starting devcontainer.")
    else:
        API_KEY = os.environ.get("API_KEY")

# base URL
BASE_URL = "https://eodhistoricaldata.com/api/"

# Format of return
FORMAT = "json"

ENDPOINTS = [
    "eod",
    "technical",
    "fundamentals"
]

###
# Helper Functions
###

def make_df(data, endpoint,call_filter=None):
    """
    Builds a cleaned DF in dependency of data_type. So its ready to go.
    
    args:
        - data (dict): json response of api call
        - endpoint (string): From which endpoint (eod etc.) is the data?
        - call_filter (string): used for endpoint "fundamentals" to generate the fitting df.
    returns:
        - pandas DataFrame object
    """
    if type(data) is list:
        if type(data[0]) is not dict:
            raise TypeError("Inner type of list input has to be dict")
    elif type(data) is dict:
        pass
    else:
        raise TypeError("Outter type of data input has to be list or dict.")


    if type(endpoint) is not str:
        raise TypeError("Content type of endpoint has to be str.")
    if endpoint not in ENDPOINTS:
        raise ValueError("Content of endpoint has to be a valid endpoint.")

    if endpoint is "eod":
        temp = pd.DataFrame.from_dict(data=data)
        temp["date"] = pd.to_datetime(arg=temp["date"], format="%Y-%m-%d")
        temp["volume"] = temp["volume"].astype("Int32")

        # Create stockstats DF
        temp = StockDataFrame.retype(temp)
        if call_filter == None:

            # Get ATR
            temp["atr"] = temp['atr']
            temp.drop(columns=["tr"], inplace=True)

            # Get ROC for last 252 and 126 trading days
            temp["close_-252_r"] = temp["close_-252_r"]
            temp["close_-126_r"] = temp["close_-126_r"]
            temp.drop(columns=["close_-1_s"], inplace=True)

            # Calculate A/D line        
            # We need Money float value first
            temp["mfv"] = (((temp["close"] - temp["low"]) - (temp["high"] - temp["close"])) + (temp["high"] - temp["low"])) * temp["volume"]

            # Now we can calculate AD-line
            temp["ad"] = 0
            for idx in range(0, len(temp.index)):
                if idx == 0:
                    temp["ad"].iloc[(idx),] = temp["mfv"].iloc[(idx),]
                else:
                    temp["ad"].iloc[(idx),] = temp["mfv"].iloc[(idx),] + temp["ad"].iloc[(idx-1),]
            # and now we can drop mfv
            temp.drop(columns=["mfv"], inplace=True)
        elif call_filter == "ECBEURUSD":
            temp["macd"] = temp["macd"]
    elif endpoint is "technical":
        temp = pd.DataFrame.from_dict(data=data)
        temp["date"] = pd.to_datetime(arg=temp["date"], format="%Y-%m-%d")
    elif endpoint is "fundamentals":
        if call_filter == "Earnings::History":
            temp = pd.DataFrame.from_dict(data=data).transpose().drop(columns=["date"])
            temp.index = pd.to_datetime(temp.index, format="%Y-%m-%d")
            for column in temp.columns:
                temp[column] = pd.to_numeric(temp[column], errors="ignore")
            temp["reportDate"] = pd.to_datetime(temp["reportDate"], format="%Y-%m-%d")
        elif call_filter == "Financials::Balance_Sheet::quarterly":
            temp = pd.DataFrame.from_dict(data=data).transpose()
            temp.index = pd.to_datetime(temp.index, format="%Y-%m-%d")
            temp.drop(columns=["date"], inplace=True)
            for column in temp.columns:
                temp[column] = pd.to_numeric(temp[column], errors="ignore")
            temp["filing_date"] = pd.to_datetime(temp["filing_date"], format="%Y-%m-%d")
            temp["book_value"] = temp["totalAssets"] - temp["totalLiab"]
        else:
            raise NotImplementedError("Given call_filter argument is not implemented or unset.")
    else:
        raise NotImplementedError("DF building for other Endpoints must be implemented.")

    return temp

###
# API functions
###

def get_data(endpoint, symbol, exchange, params, fmt=FORMAT):
    """
    Calls a specific endpoint with given symbol, exchange and params.

    args:
        - endpoint(string): Endpoint to call.
        - symbol (string): ticker symbol of stock. Can be index symbol
        - exchange (string): name of exchange. Can be index specifier.
        - params (dict): Parameter dict
        - fmt (string): Format of returned data. default is 'json'.
    returns:
        - dict in list with returned answer
    """
    check_API_KEY()
    if type(endpoint) is not str:
        raise TypeError("Content type of endpoint has to be str.")
    if endpoint not in ENDPOINTS:
        raise ValueError("Content of endpoint has to be a valid endpoint.")

    url = BASE_URL + endpoint + "/" + symbol + "." + exchange

    params["api_token"] = API_KEY
    params["fmt"] = fmt
    
    if endpoint is "eod":
        ret = requests.get(url = url, params=params).json()
    elif endpoint is "technical":
        if "period" in params:
            if type(params["period"]) is not int:
                raise TypeError("When calling 'technical' endpoint the period has to be valid integer.")
        if "function" not in params:
            raise ValueError("Parameter 'function' is required for the 'technical' endpoint.")
        ret = requests.get(url=url, params=params).json()
    elif endpoint is "fundamentals":
        params.pop("fmt")
        ret = requests.get(url=url, params=params).json()
    else:
        raise NotImplementedError("The given endpoint is not implemented.")
    return ret

