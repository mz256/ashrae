# ASHRAE Great Energy Predictor III

Collection of files for Kaggle competition by ASHRAE. Check [my Kaggle profile](https://www.kaggle.com/michelezoccali) for all kernels. The objective was stated in the competition info as:

*In this competition, you’ll develop accurate models of metered building energy usage in the following areas: chilled water, electric, hot water, and steam meters. The data comes from over 1,000 buildings over a three-year timeframe. With better estimates of these energy-saving investments, large scale investors and financial institutions will be more inclined to invest in this area to enable progress in building efficiencies.*

## Directory structure
- [notebooks](https://github.com/mz256/ashrae/tree/main/notebooks) contains some of the kernels used on Kaggle.
- [script_version](https://github.com/mz256/ashrae/tree/main/script_version) contains a version of the pipeline in python scripts, to be run either locally or in the cloud.

Please refer to the README in each folder for an in-depth description.


## Summary of results

- **LightGBM models:**

  1. Ensemble of k LightGBM models trained on different CV folds of the (preprocessed) training set. This was the highest scoring solution with a private score of 1.292.
  
  2. Ensemble of k LightGBM models per meter type, for a total of 4k models.

- **Neural network models:**

  1. DNN with entity embedding for categorical features. These are then concatenated with continuous predictors and fed to the first dense layer. We trained this learner making exclusive use of fastai, from the data loading API all the way to training and testing.
