import os
import gc
import numpy as np
import pandas as pd
import json
import time
from datetime import datetime
from contextlib import contextmanager

# load file paths
settings = json.load(open("/Users/michele/github/ashrae/settings.json"))
OUTPUT_PATH = settings["OUTPUT_PATH"]
MODEL_PATH = settings["MODEL_PATH"]
DATA_PATH = settings["DATA_PATH"]


@contextmanager
def timer(name):
    print(f'{datetime.now()} - [{name}] ...')
    t0 = time.time()
    yield
    print(f'{datetime.now()} - [{name}] done in {time.time() - t0:.0f} s\n')


def reduce_mem(df):
    result = df.copy()
    for col in result.columns:
        col_data = result[col]
        dn = col_data.dtype.name
        if not dn.startswith("datetime"):
            if dn == "object":  # the only object feature has low cardinality
                result[col] = pd.to_numeric(col_data.astype("category").cat.codes, downcast="unsigned")
            elif dn.startswith("int") | dn.startswith("uint"):
                if col_data.min() >= 0:
                    result[col] = pd.to_numeric(col_data, downcast="unsigned")
                else:
                    result[col] = pd.to_numeric(col_data, downcast='integer')
            else:
                result[col] = pd.to_numeric(col_data, downcast='float')
    return result


def add_lag_features(weather_df, window=3):
    group_df = weather_df.groupby('site_id')
    cols = ['air_temperature', 'cloud_coverage', 'dew_temperature', 'precip_depth_1_hr']
    rolled = group_df[cols].rolling(window=window, min_periods=0)
    lag_mean = rolled.mean().reset_index().astype(np.float16)
    lag_max = rolled.max().reset_index().astype(np.float16)
    lag_min = rolled.min().reset_index().astype(np.float16)
    for col in cols:
        weather_df[f'{col}_min_lag{window}'] = lag_min[col]
        weather_df[f'{col}_mean_lag{window}'] = lag_mean[col]
        weather_df[f'{col}_max_lag{window}'] = lag_max[col]
    return weather_df


def delete_bad_sitezero(X, y):
    bad_rows = X[(X.timestamp <= '2016-05-20') & (X.site_id == 0) & (X.meter == 0)].index
    X = X.drop(index=bad_rows)
    y = y.reindex_like(X)
    return X.reset_index(drop=True), y.reset_index(drop=True)
