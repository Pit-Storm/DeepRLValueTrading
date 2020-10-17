###
# Imports
###

import os
import requests
import pandas as pd

###
# Constants
###

# API_KEY
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
    "technical"
]

###
# Helper Functions
###

def make_df(data, endpoint):
    """
    Builds a cleaned DF in dependency of data_type. So its ready to go.
    
    args:
        - data (dict): json response of api call
        - endpoint (string): From which endpoint (eod etc.) is the data?
    returns:
        - pandas DataFrame object
    """
    if type(data) is not list:
        raise TypeError("Outter type of data input has to be list.")
    if type(data[0]) is not dict:
        raise TypeError("Inner type of data input has to be dict")

    if type(endpoint) is not str:
        raise TypeError("Content type of endpoint has to be str.")
    if endpoint not in ENDPOINTS:
        raise ValueError("Content of endpoint has to be a valid endpoint.")

    if endpoint is "eod":
        temp = pd.DataFrame.from_dict(data=data)
        temp["date"] = pd.to_datetime(arg=temp["date"], format="%Y-%m-%d")
        temp["volume"] = temp["volume"].astype("Int32")
        return temp
    else:
        raise NotImplementedError("DF building for other Endpoints must be implemented.")

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

    if type(endpoint) is not str:
        raise TypeError("Content type of endpoint has to be str.")
    if endpoint not in ENDPOINTS:
        raise ValueError("Content of endpoint has to be a valid endpoint.")

    url = BASE_URL + endpoint + "/" + symbol + "." + exchange

    params["api_token"] = API_KEY
    params["fmt"] = fmt
    
    if endpoint is "eod":
        ret = requests.get(url = url, params=params).json()
    if endpoint is "technical":
        if "period" in params:
            if type(params["period"]) is not int:
                raise TypeError("When calling 'technical' endpoint the period has to be valid integer.")
        if "function" not in params:
            raise ValueError("Parameter 'function' is required fot the 'technical' endpoint.")
        ret = requests.get(url=url, params=params).json()

    return ret

def get_ti(symbol, exchange, params, fmt=FORMAT):
    """

    """