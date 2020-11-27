import pandas as pd
from dateutil.relativedelta import relativedelta
from pathlib import Path

def load_data(path: Path):
    """
    Creates a pandas Dataframe from file provided by path.
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

def train_val_test_split(df, years=4):
    """
    Splits df into train, val and test. Val and Test is slice of years.

    params:
        - df: Pandas DataFrame with DateTimeIndex name 'date'.
        - years (int): default 4. What lenght should val and test have?

    returns:
        - train, val, test (tuple): slices of df.
    """
    dates = df.index.get_level_values(level="date")
    # creating test df
    test_start = dates.max() - relativedelta(years=years)
    test = df[dates > test_start]
    # creating val df
    val_start = test_start - relativedelta(years=years)
    val = df[(dates > val_start) & (dates < test_start)]
    # creating train df
    train = df[(dates < val_start)]
    return (train, val, test)
