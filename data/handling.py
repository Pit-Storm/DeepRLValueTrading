# %%
import pandas as pd

def load_data(path):
    """
    Creates a pandas Dataframe with from file provided with path.
    Set index to date and symbol.

    params:
        - path: file path to a .csv file
    
    returns:
        - pandas.DataFrame object
    """
    df =    (pd.read_csv(path, parse_dates=["date"])
                .set_index(["date","symbol"])
                .drop(columns=["Unnamed: 0"])
            )
    return df

# %%
if __name__ == "__main__":
    df = load_data("stocksdata_all.csv")
    dates = df.index.get_level_values(level="date")
    date = dates[0]
    df.head()
    df[dates == date]["close"].values.tolist()
    indicators = df.columns.tolist()
    indicator_values = [item for indicator in indicators for item in df[dates == date][indicator].values.tolist()]
    indicator_values
    df.columns
# %%
