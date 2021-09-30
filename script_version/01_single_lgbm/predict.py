import gc
import numpy as np
import pandas as pd
import argparse
import pickle
import seaborn as sns
from tqdm import tqdm
from utils import (
    DATA_PATH, MODEL_PATH, OUTPUT_PATH, timer)

parser = argparse.ArgumentParser(description='')

parser.add_argument('--debug', action='store_true', help='Run in debug mode')

if __name__ == "__main__":

    args = parser.parse_args()
    debug = args.debug

    with open(f'{MODEL_PATH}/lgb_3fold.pickle', mode='rb') as f:
        [models] = pickle.load(f)

    with timer('Loading preprocessed test data'):
        # for lgb use version with lag features, as the training is not too expensive
        X_test = pd.read_hdf(f"{DATA_PATH}/preprocessed/lgb_with_lag.h5", key="test")
        row_ids = X_test.row_id
        X_test = X_test.drop(columns='row_id')

        gc.collect()
        X_test.info()


    with timer('Inferring in batches'):
        n_iterations = 20
        batch_size = len(X_test) // n_iterations

        preds = []
        for i in tqdm(range(n_iterations)):
            start = i * batch_size
            fold_preds = [np.expm1(model.predict(X_test.iloc[start:start + batch_size], 
                                                 num_iteration=model.best_iteration)) for model in models]
            preds.extend(np.mean(fold_preds, axis=0))

        del X_test
        gc.collect()

    with timer('Writing to submission file'):
        submission = pd.DataFrame({'row_id':row_ids, 'meter_reading':np.clip(preds, 0, a_max=None)})
        if not debug:
            submission.to_csv(f'{OUTPUT_PATH}/submission.csv', index=False)
