# Forecast

This repository aims at using some recent machine learning methods to solve a forecasting problem. For each model implemented, we try to follow the steps below

- **Data preparation:** Inputs are pre-processed, normalized and key features extracted
- **Model training:** Model implementation and parameters' fitting
- **Evaluation:**  Model testing to ensure satisfactory model performance 

Some tests are added to ensure the correct model implementation and expected learning behaviour

## Quick Start

Use the commands below to run the code 
```
# Prepare environment
git clone https://github.com/anonymot23/Forecast.git
cd Forecast

# Sample model run
python tests.BASELINE.test_baseline_simple
```

## Structure

The code is decomposed into three folders

- **data:** containing the input data
- **src:** containing necessary code for models' implementation
- **tests:** containing tests and sample models' runs


## Data 

A standard dataset is used to test models. It contains information about monthly totals of a US airline passengers from 1949 to 1960.

## Models comparison

The results below hold only for our implementation. Additional model improvements, new features, and different data can produce different values.

| Model 				| MSE  | RMSE | MAE  | MAPE | SMAPE | SWAPE |
| ---   				| ---  | ---  | ---  | ---  | ---   | ---   |
| Baseline              | 0.12 | 0.35 | 0.31 | 0.31 | 0.20  | 0.32  | 
| Linear Regression     | 0.08 | 0.28 | 0.26 | 0.28 | 0.17  | 0.27  |
| Exponential smoothing | 0.08 | 0.28 | 0.25 | 0.28 | 0.17  | 0.26  |
| LightGBM              | 0.25 | 0.50 | 0.36 | 0.50 | 0.21  | 0.47  |
| XGBoost 				| 0.12 | 0.35 | 0.33 | 0.33 | 0.20  | 0.34  |
| Random Forest 		| 0.13 | 0.36 | 0.35 | 0.35 | 0.21  | 0.37  | 
| SARIMAX 				| 0.01 | 0.12 | 0.10 | 0.14 | 0.08  | 0.12  | 
| TBATS 				| 0.08 | 0.28 | 0.25 | 0.28 | 0.17  | 0.29  | 
| Prophet 				| 0.12 | 0.35 | 0.33 | 0.34 | 0.22  | 0.35  | 
| NBEATS 				| 0.01 | 0.07 | 0.06 | 0.11 | 0.06  | 0.11  | 
| NHITS 				| 0.01 | 0.06 | 0.06 | 0.10 | 0.05  | 0.10  |
| TFT 					| 0.01 | 0.06 | 0.05 | 0.09 | 0.05  | 0.09  |



## Tests