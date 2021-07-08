# ASHRAE Great Energy Predictor III

Collection of files for Kaggle competition by ASHRAE. The objective was stated in the competition info as:

*In this competition, you’ll develop accurate models of metered building energy usage in the following areas: chilled water, electric, hot water, and steam meters. The data comes from over 1,000 buildings over a three-year timeframe. With better estimates of these energy-saving investments, large scale investors and financial institutions will be more inclined to invest in this area to enable progress in building efficiencies.*

The folder `notebooks` contains some of the kernels used on Kaggle, while the remaining folders contain a version of the pipeline in .py scripts to be run either locally or remotely.

## LGBM

The highest scoring solution consists of an ensemble of k LGBMs trained on different CV folds of the whole (preprocessed) training set. All k models share the same set of hyperparameters. This is contained in the `kfold` folder.

Another solution comprises k LGBMs trained on CV folds for each energy meter, resulting in a total of 4k models. Code in the `notebooks` folder.

## Neural Net with Categorical Embeddings

Another solution involves training a neural network with the fast.ai API. This neural net feeds the categorical features to Embedding layers and concatenates the resulting embedding vectors with the continuous features before feeding to the first linear layer. Code in `notebooks`.
