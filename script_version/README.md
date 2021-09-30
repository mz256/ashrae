A collection of scripts to run the analysis pipelines more seamlessly and to move them on the cloud when needed (_work in progress_).

## Set-up

This code is best run in the environment provided by `environment.yml`. Please consider beginning by running the following snippet of code:
```
cd ashrae/script_version
conda env create
conda activate env-ashrae
```

## LightGBM models

[01_single_lgbm](https://github.com/mz256/ashrae/tree/main/script_version/01_kfold_lgbm): Preprocessing, feature engineering, training k LightGBM models with cross-validation, inference with simple ensemble. We provide `bash run.sh` in order to easily reproduce the whole pipeline with given parameters.

## Deep learning models

[02_dnn](https://github.com/mz256/ashrae/tree/main/script_version/02_dnn): To be completed.
