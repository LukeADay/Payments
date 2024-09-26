# Predicting Executed Payments :bar_chart:

This project involves the analysis of a payments dataset, where interest lies in optimising the conversion of payments to an executed status. Payments can go through various stages between initiated, to failing or succeeding (executed) and many in between.

A machine learning modelling approach was employed using a `h2o` GBM model, to predict the success of a payment executing based on various features.

## Structure of Repository :clipboard:

This repository is structured as follows:

```
├── README.md
├── config
│   └── h2o_env.yml
├── data
│   └── truelayer_data_sc_test_data_set.csv
├── evaluation
│   └── performance.csv
├── log
├── models
│   └── gbm_grid_1_model_4
├── notebooks
│   ├── EDA.ipynb
│   └── Summary.ipynb
├── scripts
│   ├── __init__.py
│   ├── main.py
│   └── setup_env.sh
├── src
│   ├── __init__.py
│   ├── __pycache__
│   │   └── __init__.cpython-312.pyc
│   ├── data_handler
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-312.pyc
│   │   │   └── data_loader.cpython-312.pyc
│   │   └── data_loader.py
│   ├── modelling
│       ├── __init__,py
│       ├── __pycache__
│       │   └── model.cpython-312.pyc
│       └── model.py
```

* `scripts/main.py` runs the data processing and modelling pipeline - saving various model performance metrics in the `evaluation` folder and the serialized model in the `models` folder.
  + This is done by calling two classes, `src/data_handler/data_loader.py` and `src/modelling/model.py`.
*  `EDA.ipynb` includes exploratory data analysis, assessing data quality, determining univariate associations and performing feature engineering.
* `summary.ipynb` produces a **summary of the analysis**, includes details behind feature engineering, model selection (including justification), training, evaluation, deployment strategy and intepretation.

  
## Usage - how to run the pipeline :mag:

1. [x] Begin by cloning the repository and place the data file in the `data` folder.
2. [x] Open terminal and navigate to the `scripts` folder with `cd scripts` and run the shell script `setup_env.sh` to set up the Conda :snake: environment by running either `bash setup_env.sh` or `./setup_env.sh` in the terminal.
3. [x] Navigate back to the parent directory `Payments` with `cd ..`.
4. [x] Run the pipeline with `PYTHONPATH=. python scripts/main.py` from the `Payments` folder. The following help file shows that `predictors` is the only argument this script takes, which consists of the features that model is trained on. This can be left blank and the default set is used.

```
usage: main.py [-h] [--predictors PREDICTORS [PREDICTORS ...]]

Run the data processing and modelling pipeline and fit h2o gbm model

options:
  -h, --help            show this help message and exit
  --predictors PREDICTORS [PREDICTORS ...]
                        List of predictor variables (space-separated)
```

## Outputs

The pipeline saves 2 models: 
* A H2O Binary model `gbm_grid_1_model_4`, which can be loaded but requires the same version of H2O.
* A MOJO model (model object optimised) - this should be used for production as it doesn't require a specific version of H2O. This can be used for real-time scoring in production.

The `Summary.ipynb` file loads the trained H2O binary model from the `models` directory with the processed data and produces **variable importance**, **shap summary plots** and **partial dependence plots**. This file also contains details behind the various stages of the analysis.

