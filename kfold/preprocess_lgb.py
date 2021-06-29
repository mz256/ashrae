#!/usr/bin/env python

import os
import gc
import numpy as np
import pandas as pd
import datetime
import argparse
from utils import (
    DATA_PATH, timer, reduce_mem, add_lag_features,
    delete_bad_sitezero)

parser = argparse.ArgumentParser(description="")

parser.add_argument("--add_lag", action="store_true", help="If True add lag features")


def load_data(source='train'):
    assert source in ['train', 'test']
    df = pd.read_csv(f'{DATA_PATH}/{source}.csv', parse_dates=['timestamp'])
    return reduce_mem(df)


def load_building():
    df = pd.read_csv(f'{DATA_PATH}/building_metadata.csv').fillna(-1)
    return reduce_mem(df)


def load_weather(source='train', fix_timezone=True, impute=True, add_lag=True):
    assert source in ['train', 'test']
    df = pd.read_csv(f'{DATA_PATH}/weather_{source}.csv', parse_dates=['timestamp'])
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
        x = df.drop('meter_reading', axis=1)
        y = np.log1p(df.meter_reading)  # log-transform of target
        return x, y
    elif source == 'test':
        return df


def extract_temporal(X):
    X['hour'] = X.timestamp.dt.hour
    X['weekday'] = X.timestamp.dt.weekday
    # month and year cause overfit, could try other (holiday, business, etc.)
    return reduce_mem(X)


if __name__ == '__main__':
    """
    python preprocess.py
    python preprocess.py --add_lag
    """

    args = parser.parse_args()

    lag = False
    if args.add_lag:
        lag = True

    # TRAINING DATASET
    with timer("Loading and processing training data"):
        X_train, y_train = merged_dfs(add_lag=lag)

        # delete bogus site 0 readings and extract time features
        X_train, y_train = delete_bad_sitezero(X_train, y_train)
        X_train = extract_temporal(X_train)

        # remove timestamp and other unimportant features
        to_drop = ['timestamp', 'sea_level_pressure', 'wind_direction', 'wind_speed']
        X_train.drop(to_drop, axis=1, inplace=True)
        gc.collect()

        # concatenate for compact save
        df_train = pd.concat([X_train, y_train], axis=1)
        del X_train, y_train
        gc.collect()

        df_train.info()

    with timer("Saving training data"):
        if args.add_lag:
            fname = "lgb_with_lag.h5"
        else:
            fname = "lgb_no_lag.h5"

        # save to HDF5 to preserve dtypes (overwrite previous version)
        df_train.to_hdf(f'{DATA_PATH}/preprocessed/{fname}', index=False, key='train', mode='w')
        del df_train
        gc.collect()

    # TEST DATASET
    with timer("Loading and processing test data"):
        X_test = merged_dfs(source='test', add_lag=lag)
        X_test = extract_temporal(X_test)
        X_test.drop(columns=['timestamp'] + to_drop, inplace=True)
        gc.collect()

        X_test.info()

    with timer("Saving test data"):
        # now simply append dataset to the same file
        X_test.to_hdf(f'{DATA_PATH}/preprocessed/{fname}', index=False, key='test', mode='a')
