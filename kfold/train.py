
import sys
import gc
import psutil
import warnings
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from utils import (
            DATA_PATH, MODEL_PATH, timer
)

parser = argparse.ArgumentParser(description='')

parser.add_argument('--debug', action='store_true', help='Run in debug mode')

parser.add_argument('--n_splits', type=int, default=3,
                    help='Number of cross-validation folds')


def rf_wrapper(Xt, yt, Xv, yv, fold=-1):
    model = RandomForestRegressor(n_jobs=-1, n_estimators=40,
                                  max_samples=200000, max_features=0.5,
                                  min_samples_leaf=5, oob_score=False).fit(Xt, yt)
    print(f'Training fold {fold}...')

    score_train = np.sqrt(mean_squared_error(model.predict(Xt), yt))
    oof = model.predict(Xv)
    score = np.sqrt(mean_squared_error(oof, yv))
    print(f'Fold {fold}: training RMSLE: {score_train},   validation RMSLE: {score}\n')
    return model, oof, score


def lgbm_wrapper(xt, yt, xv, yv, fold=-1):
    dset = lgb.Dataset(xt, label=yt, categorical_feature=cat_features)
    dset_val = lgb.Dataset(xv, label=yv, categorical_feature=cat_features)

    if debug:
        num_leaves = 5; boost_rounds = 100; early = 10
    else:
        num_leaves = 5; boost_rounds = 100; early = 10

    params = {
        "objective": "regression",
        "boosting": "gbdt",
        "num_leaves": num_leaves,
        "learning_rate": 0.04,
        "feature_fraction": 0.7,
        "subsample": 0.4,
        "metric": "rmse",
        "seed": 42,
        "n_jobs": -1,
        "verbose": -1
    }

    print(f'Fold {fold}')

    # filter some known warnings (open issue at https://github.com/microsoft/LightGBM/issues/3379)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "categorical_column in param dict is overridden")
        warnings.filterwarnings("ignore", "Overriding the parameters from Reference Dataset")
        model = lgb.train(params,
                          train_set=dset,
                          num_boost_round=boost_rounds,
                          valid_sets=[dset, dset_val],
                          verbose_eval=200,
                          early_stopping_rounds=early,
                          categorical_feature=cat_features)

    oof = model.predict(xv, num_iteration=model.best_iteration)
    score = np.sqrt(mean_squared_error(yv, oof))
    print(f'Fold {fold} validation RMSLE: {score}\n')
    return model, oof, score


def perform_cv(wrapper, n_splits=3):
    kf = KFold(n_splits=n_splits, shuffle=False)

    models = []
    scores = []
    oof_total = np.zeros(X_train.shape[0])

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
        Xt, yt = X_train.iloc[train_idx], y_train[train_idx]
        Xv, yv = X_train.iloc[val_idx], y_train[val_idx]
        model, oof, score = wrapper(Xt, yt, Xv, yv, fold)

        models.append(model)
        scores.append(score)
        oof_total[val_idx] = oof

    print('Training completed.')
    print(f'> Mean RMSLE across folds: {np.mean(scores)}, std: {np.std(scores)}')
    print(f'> OOF RMSLE: {np.sqrt(mean_squared_error(y_train, oof_total))}')
    return models, scores, oof_total


if __name__ == "__main__":

    args = parser.parse_args()
    debug = args.debug
    n_splits = args.n_splits

    with timer('Loading preprocessed training data'):
        # for lgb use version with lag features, as the training is not too expensive
        df_train = pd.read_hdf(f"{DATA_PATH}/preprocessed/lgb_with_lag.h5", key="train")
        y_train = df_train['meter_reading']
        X_train = df_train.drop(columns='meter_reading')
        del df_train
        gc.collect()

        X_train.info()

    #with timer(f'Training random forest (for baseline) with {n_splits}-fold cross-validation'):
    #    _, _, _ = perform_cv(rf_wrapper, n_splits=n_splits)

    with timer(f'Training LightGBM with {n_splits}-fold cross-validation'):
        cat_features = ['building_id', 'meter', 'site_id', 'primary_use', 'hour', 'weekday']
        models, _, _ = perform_cv(lgbm_wrapper, n_splits=n_splits)

    with timer('Saving models'):
        if not debug:
            import pickle
            with open(f'{MODEL_PATH}/lgb_{n_splits}fold.pickle', mode='wb') as f:
                pickle.dump([models], f)

        del models
