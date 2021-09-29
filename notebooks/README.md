These are some of the notebooks used on Kaggle. For a complete list, please see [https://www.kaggle.com/michelezoccali](kaggle.com/michelezoccali). They mainly differ in the models which were trained, while the preprocessing and feature engineering phases were largely similar.

## Preprocessing

- **Import:** Three dataframes are imported (main dataset with meter readings, building metadata and weather data). 
- **Downcasting:** All are carefully typecasted to minimise memory usage, given the large size of the datasets. 
- **Data cleaning:** Weather data are corrected for time-zone misalignment and missing values are imputed with linear interpolation. Dataframes are merged and outliers coming from faulty meter readings in 1 out of 16 sites are removed.

## Feature engineering

- **Date-time features:** Common to all notebooks, we extracted hour and weekday from timestamps and dropped unimportant features. 
- **Lag features:** The LGBM models were additionally fed a host of lagged versions of weather features whose temporal persistence could strongly drive energy consumption over multiple timestamps (e.g. air temperature, cloud coverage). On the other hand, the `tabularNN` was not fed lag features for memory constraints due to the overhead of fastai objects.

## Model training and inference

1. `single_lgbm` trains a gradient boosted machine for all meter types with k-fold cross validation. Inference is performed by bagging the k models, which greatly improves model performance over that of any single model (even if re-trained on all training data).
2. `meter_specific_lgbm` trains 4 gradient boosted machines, one for each meter type, in an attempt to grasp meter-specific effects. Inference on the relevant subset of the data is still performed by bagging the k CV versions of that meter's model.
3. `tabularNN` trains a DNN optimised for tabular data. An embedding layer turns categorical variables into numerical feature vectors, these are concatenated with the continuous features and finally fed to a fully-connected network.




