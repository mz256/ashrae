#!/usr/bin/env python

import os
import gc
import numpy as np
import pandas as pd
import datetime

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

def _delete_bad_sitezero(X, y):
    bad_rows = X[(X.timestamp <= '2016-05-20') & (X.site_id == 0) & (X.meter == 0)].index
    X = X.drop(index=bad_rows)
    y = y.reindex_like(X)
    return X.reset_index(drop=True), y.reset_index(drop=True)

def _extract_temporal(X, train=True):
    X['hour'] = X.timestamp.dt.hour
    X['weekday'] = X.timestamp.dt.weekday
    if train:
        # include month to create validation set, to be deleted before training
        X['month'] = X.timestamp.dt.month
    # month and year cause overfit, could try other (holiday, business, etc.)
    return reduce_mem(X)

def load_data(source='train'):
    assert source in ['train', 'test']
    df = pd.read_csv(f'{path}/{source}.csv', parse_dates=['timestamp'])
    return reduce_mem(df)


def load_building():
    df = pd.read_csv(f'{path}/building_metadata.csv').fillna(-1)
    return reduce_mem(df)


def load_weather(source='train', fix_timezone=True, impute=True, add_lag=True):
    assert source in ['train', 'test']
    df = pd.read_csv(f'{path}/weather_{source}.csv', parse_dates=['timestamp'])
    if fix_timezone:
        offsets = [5, 0, 9, 6, 8, 0, 6, 6, 5, 7, 8, 6, 0, 7, 6, 6]
        offset_map = {site: offset for site, offset in enumerate(offsets)}
        df.timestamp = df.timestamp - pd.to_timedelta(df.site_id.map(offset_map), unit='h')
    if impute:
        site_dfs = []
        for site in df.site_id.unique():
            if source == 'train':
                new_idx = pd.date_range(start='2016-1-1', end='2016-12-31-23', freq='H')
            else:
                new_idx = pd.date_range(start='2017-1-1', end='2018-12-31-23', freq='H')
            site_df = df[df.site_id == site].set_index('timestamp').reindex(new_idx)
            site_df.site_id = site
            for col in [c for c in site_df.columns if c != 'site_id']:
                site_df[col] = site_df[col].interpolate(limit_direction='both', method='linear')
                site_df[col] = site_df[col].fillna(df[col].median())
            site_dfs.append(site_df)
        df = pd.concat(site_dfs)
        df['timestamp'] = df.index
        df = df.reset_index(drop=True)

    if add_lag:
        df = add_lag_features(df, window=3)

    return reduce_mem(df)


def merged_dfs(source='train', fix_timezone=True, impute=True, add_lag=False):
    df = load_data(source=source).merge(load_building(), on='building_id', how='left')
    df = df.merge(load_weather(source=source, fix_timezone=fix_timezone, impute=impute, add_lag=add_lag),
                  on=['site_id', 'timestamp'], how='left')
    if source == 'train':
        X = df.drop('meter_reading', axis=1)
        y = np.log1p(df.meter_reading)  # log-transform of target
        return X, y
    elif source == 'test':
        return df