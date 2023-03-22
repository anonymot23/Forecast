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

The results below hold only for our implementation. Additional model improvements and features incorporation can produce different values.

| Model 				| MSE | RMSE | MAPE | SMAPE | SWAPE |
| ---   				| --- | ---  | ---  | ---   | ---   |
| Baseline              |     |      |      |       |       |
| Linear Regression     |     |      |      |       |       |
| Exponential smoothing |     |      |      |       |       |
| LightGBM              |     |      |      |       |       |
| XGBoost 				|     |      |      |       |       |
| Random Forest 		|     |      |      |       |       |
| SARIMAX 				|     |      |      |       |       | 
| TBATS 				|     |      |      |       |       |
| Prophet 				|     |      |      |       |       | 
| NBEATS 				|     |      |      |       |       | 
| NHITS 				|     |      |      |       |       | 
| TFT 					|     |      |      |       |       | 



## Tests