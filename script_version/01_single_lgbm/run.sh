#!/bin/bash
python preprocess_lgb.py --add_lag &> preprocess_lgb.log
python train.py --n_splits 3 &> train.log
python predict.py &> predict.log
