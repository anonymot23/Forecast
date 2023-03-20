import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import cmdstanpy
# cmdstanpy.install_cmdstan()
# cmdstanpy.install_cmdstan(compiler=True)# only valid on Windows

import matplotlib.pyplot as plt


#from tqdm import tqdm_notebook as tqdm

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import NHiTSModel
from darts.metrics import mape
from darts.utils.statistics import check_seasonality, plot_acf
from darts.datasets import AirPassengersDataset, IceCreamHeaterDataset
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.utils.likelihood_models import QuantileRegression
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller


import torch 
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import warnings

warnings.filterwarnings("ignore")
import logging

logging.disable(logging.CRITICAL)


# simple test : 30 min
## create project more/less clean containing multiple model 
### folder for data  ok
### file for data preparation  ok 
### file for model ok 
### eventually file for utils 
### file to generate forecast ok
### main file combining everything ok
### file for global parameters ok 
### folder for testing ok 


# # parameters
# lag = 12

# # collect and load data 
# data_folder = "../../data"
# filename = 'AirPassengers.csv'
# dataAir = pd.read_csv(f"{data_folder}/{filename}")


# # prepare/pre-process data 
# dataAir['Month'] = pd.to_datetime(dataAir['Month'])
# dataAir.set_index(dataAir['Month'], inplace=True)
# scaler = StandardScaler()
# passengers = dataAir['#Passengers'].values.reshape((-1, 1))
# scaler.fit(passengers)
# dataAir['#Passengers'] = scaler.transform(passengers)

# ## split train/test data 
# timeTrain = [pd.to_datetime('1950-01-01') , pd.to_datetime('1957-12-01')]
# timeTest = [pd.to_datetime('1958-01-01') , pd.to_datetime('1959-01-01')]

# dataTrain = dataAir[timeTrain[0]: timeTrain[1]]
# indexTrain = dataTrain.index
# valTrain = dataTrain['#Passengers'].values
# yTrain = pd.Series(valTrain, indexTrain)

# dataTest = dataAir[timeTest[0]: timeTest[1]]
# indexTest = dataTest.index
# valTest = dataTest['#Passengers'].values
# yTest = pd.Series(valTest, indexTest)

# # build and fit model 
# minValTrain = valTrain.min()-0.1
# modelES = ExponentialSmoothing(yTrain-minValTrain, 
#                                seasonal_periods=lag, 
#                                trend='add',
#                                seasonal='add',
#                                use_boxcox=True,
#                                initialization_method='estimated')
# modelESFitted = modelES.fit()

# # test/evaluate the model  
# lenTest = len(yTest) 
# predTest = modelESFitted.forecast(lenTest) + minValTrain
# error = mean_squared_error(predTest, yTest)
# print(error)

# # plot values 
# plt.plot(predTest)
# plt.plot(yTest)
# plt.show()


# scaler = StandardScaler()
# scaler.fit(data)
# scaled = scaler.transform(data)
# print(scaled)

# # for inverse transformation
# inversed = scaler.inverse_transform(scaled)
# print(inversed)

if __name__ == "__main__":
    # parameters 
    windowSize = 3
    shift = 1
    lag = 12
    
    
    # data collection and loading data 
    data_folder = "../../../data"
    filename = 'AirPassengers.csv'
    dataAir = pd.read_csv(f"{data_folder}/{filename}")
    
    
    # data pre-processing/preparation # data already preprocessed here
    dataAir['Month'] = pd.to_datetime(dataAir['Month'])
    dataAir.set_index(dataAir['Month'], inplace=True)
    scaler = StandardScaler()
    passengers = dataAir['#Passengers'].values.reshape((-1,1))
    scaler.fit(passengers)
    dataAir['#Passengers'] = scaler.transform(passengers)
    
    ## prepare features
    dataAir['Before1'] = dataAir['#Passengers'].shift(shift)
    dataAir['BeforeLag'] = dataAir['#Passengers'].shift(lag)
    # dataAir['MA'] = dataAir['#Passengers'].rolling(windowSize).mean()
    # dataAir['MALag'] = dataAir['MA'].shift(lag)
    
    ## test/train split
    timeTrain = [pd.to_datetime('1950-01-01') , pd.to_datetime('1957-12-01')]
    timeTest = [pd.to_datetime('1958-01-01') , pd.to_datetime('1959-01-01')]
    
    dataTrain = dataAir[timeTrain[0]: timeTrain[1]]
    dataTest = dataAir[timeTest[0]: timeTest[1]]
    
    # reformat data
    yTrain = dataTrain['#Passengers'].values
    df_train = pd.DataFrame.from_dict({"ds": dataTrain.index, "y": yTrain.flatten()})

    yTest = dataTest['#Passengers'].values
    df_test = pd.DataFrame.from_dict({"ds": dataTest.index, "y": yTest.flatten()})

    
    # ## remove NA
    # filterNA = dataTrain['MALag'] == dataTrain['MALag']
    # dataTrain = dataTrain[filterNA]
    
    # ## split features and targets
    # xTrain = dataTrain[['MALag']].values
    # yTrain = dataTrain['#Passengers'].values
    
    # xTest = dataTest[['MALag']].values
    # yTest = dataTest['#Passengers'].values
    
    
    # # build and fit model 
    
    
    # modelLR = LinearRegression()
    # # modelLR.fit(xTrain, yTrain)
    # modelLR.fit(xTest, yTest)
    
    
    # # test/evaluate the model  
    # predTest = modelLR.predict(xTest)
    # error = mean_squared_error(predTest, yTest)
    # print(error)
    
    
    # error1 = mean_squared_error(xTest*1.1, yTest)
    # print(error1)
    
    
    # ma = dataAir['#Passengers'].rolling(windowSize).mean()
    
    # pred = dataAir['#Passengers'].copy()
    # pred.iloc[windowSize+lag:] = ma.iloc[windowSize:-lag].values * 1.1 # try to estimate this parameter
    # pred.iloc[:windowSize+lag] = np.nan
    
    
    # time_test = [pd.to_datetime('1958-01-01'), pd.to_datetime('1959-01-01')]
    
    # y_test = dataAir['#Passengers'][time_test[0]: time_test[1]].values
    # pred_test =  pred[time_test[0]: time_test[1]].values
    # error2 = mean_squared_error(y_test, pred_test)
    
    # print(error2)
    
    # plt.plot(y_test)
    # plt.plot(pred_test)
    # plt.show()
    
    
    # plt.plot(y_test)
    # plt.plot(predTest)
    # plt.show()
    
    
    # plt.plot(dataAir['#Passengers'])
    # plt.plot(dataAir['MALag']*1.1)
    # plt.show()
    
    
    # timeTrain = [pd.to_datetime('1950-01-01'), pd.to_datetime('1957-12-01')]
    # timeTest = [pd.to_datetime('1958-01-01'), pd.to_datetime('1959-01-01')]
    # lag = 12
    
    # scaler = StandardScaler()
    # passengers = dataAir['#Passengers'].values.reshape((-1,1))
    # scaler.fit(passengers)
    # dataAir['#Passengers'] = scaler.transform(passengers)
    # dataAir['Before1'] = dataAir['#Passengers'].shift(1)
    # dataAir['BeforeLag'] = dataAir['#Passengers'].shift(lag)
    
    
    # dataTrain = dataAir[timeTrain[0]: timeTrain[1]]
    # X_train = dataTrain[['Before1', 'BeforeLag']].fillna(0).values
    # y_train = dataTrain['#Passengers'].values
    # X_train_rescaled = StandardScaler().fit_transform(X_train)
    # y_train_rescaled = StandardScaler().fit_transform(y_train.reshape(-1,1))
    
    # dataTest = dataAir[timeTest[0]: timeTest[1]]
    # X_test = dataTest[['Before1', 'BeforeLag']].fillna(0).values
    # y_test = dataTest['#Passengers'].values
    # X_test_rescaled = StandardScaler().fit_transform(X_test)
    # y_test_rescaled = StandardScaler().fit_transform(y_test.reshape(-1,1))
    
    
    # # prepare inputs bis
    # df_train = pd.DataFrame.from_dict({"ds": dataTrain.index, "y": y_train_rescaled.flatten()})
    # df_test = pd.DataFrame.from_dict({"ds": dataTest.index, "y": y_test_rescaled.flatten()})
    
    
    # # train model
    # model_proph = Prophet(changepoint_prior_scale=0.1, seasonality_prior_scale=1)
    # # model_proph = Prophet()
    # model_proph.fit(df_train)
    # # model_proph.fit(df_test)
    # y_res_proph = model_proph.predict(df_test[["ds"]])
    # pred_test = y_res_proph['yhat']
    
    # error2 = mean_squared_error(yTest, pred_test)
    # print(error2)
    
    # plt.plot(yTest)
    # plt.plot(pred_test)
    # plt.show()
    
    
    # # plt.plot(yTest)
    # # plt.plot(pred_test)
    # # plt.show()
    

    # before starting, we define some constants
    num_samples = 200
    
    figsize = (9, 6)
    lowest_q, low_q, high_q, highest_q = 0.01, 0.1, 0.9, 0.99
    label_q_outer = f"{int(lowest_q * 100)}-{int(highest_q * 100)}th percentiles"
    label_q_inner = f"{int(low_q * 100)}-{int(high_q * 100)}th percentiles"
    
    
    # Read data
    series = AirPassengersDataset().load()
    
    # we convert monthly number of passengers to average daily number of passengers per month
    series = series / TimeSeries.from_series(series.time_index.days_in_month)
    series = series.astype(np.float32)
    
    # Create training and validation sets:
    training_cutoff = pd.Timestamp("19571201")
    train, val = series.split_after(training_cutoff)
    
    # Normalize the time series (note: we avoid fitting the transformer on the validation set)
    transformer = Scaler()
    train_transformed = transformer.fit_transform(train)
    val_transformed = transformer.transform(val)
    series_transformed = transformer.transform(series)
    
    # create year, month and integer index covariate series
    covariates = datetime_attribute_timeseries(series, attribute="year", one_hot=False)
    covariates = covariates.stack(
        datetime_attribute_timeseries(series, attribute="month", one_hot=False)
    )
    covariates = covariates.stack(
        TimeSeries.from_times_and_values(
            times=series.time_index,
            values=np.arange(len(series)),
            columns=["linear_increase"],
        )
    )
    covariates = covariates.astype(np.float32)
    
    # transform covariates (note: we fit the transformer on train split and can then transform the entire covariates series)
    scaler_covs = Scaler()
    cov_train, cov_val = covariates.split_after(training_cutoff)
    scaler_covs.fit(cov_train)
    covariates_transformed = scaler_covs.transform(covariates)
    print(series)


    # create training and validation sets 
    training_cutoff = pd.Timestamp("19571201")
    train, val = series.split_after(training_cutoff)
    cov_train, cov_val = covariates.split_after(training_cutoff)
    
    # Normalize time series 
    scaler = Scaler()
    scalerCov = Scaler()
    train = scaler.fit_transform(train)
    val = scaler.transform(val)
    cov_train = scalerCov.fit_transform(cov_train)

    # Build model 
    # some fixed parameters that will be the same for all models
    MAX_N_EPOCHS = 300
    MAX_SAMPLES_PER_TS = 180
    input_chunk_length = 64
    output_chunk_length = 24
    num_stacks = 3# 3
    num_blocks = 3# 3
    num_layers = 3# 3
    layer_exp = 7# 10
    dropout = 0.1
    lr = 1e-3
        
    # throughout training we'll monitor the validation loss for early stopping
    early_stopper = EarlyStopping("val_loss", min_delta=0.0001, patience=2, verbose=True)
    callbacks = [early_stopper]
    
    # detect if a GPU is available
    pl_trainer_kwargs = {"callbacks": callbacks}
    num_workers = 1
    
    # model 
    modelNhits = NHiTSModel(
        input_chunk_length = input_chunk_length,
        output_chunk_length = output_chunk_length,
        num_stacks = num_stacks,
        num_blocks = num_blocks,
        num_layers = num_layers,
        layer_widths = 2**layer_exp,
        dropout = dropout,
        n_epochs = MAX_N_EPOCHS,
        batch_size = 128,
        add_encoders = None,
        likelihood = None,
        loss_fn = torch.nn.MSELoss(),
        random_state = 42,
        pl_trainer_kwargs = pl_trainer_kwargs,
        force_reset = True,
        save_checkpoints = True
    )
    
    # fit model
    modelNhits.fit(
        series =  train,
        val_series = train,
        past_covariates = cov_train,
        val_past_covariates = cov_train,
        max_samples_per_ts = MAX_SAMPLES_PER_TS,
        num_loader_workers = num_workers
    )



    timeTest = [pd.to_datetime('1958-01-01'), pd.to_datetime('1959-01-01')]
    nbSamples = 200
    
    y_test = val.pd_dataframe()['#Passengers'][timeTest[0]: timeTest[1]].values
    pred_nhits = modelNhits.predict(n=len(y_test) , num_samples = nbSamples).pd_dataframe().mean(axis=1)
    error = mean_squared_error(y_test, pred_nhits)
    
    print(error)

    plt.plot(y_test)
    plt.plot(pred_nhits.values)

    # # default quantiles for QuantileRegression
    # quantiles = [
    #     0.01,
    #     0.05,
    #     0.1,
    #     0.15,
    #     0.2,
    #     0.25,
    #     0.3,
    #     0.4,
    #     0.5,
    #     0.6,
    #     0.7,
    #     0.75,
    #     0.8,
    #     0.85,
    #     0.9,
    #     0.95,
    #     0.99,
    # ]
    # input_chunk_length = 24
    # forecast_horizon = 12
    # my_model = TFTModel(
    #     input_chunk_length=input_chunk_length,
    #     output_chunk_length=forecast_horizon,
    #     hidden_size=64,
    #     lstm_layers=1,
    #     num_attention_heads=4,
    #     dropout=0.1,
    #     batch_size=16,
    #     n_epochs=300,
    #     add_relative_index=False,
    #     add_encoders=None,
    #     likelihood=QuantileRegression(
    #         quantiles=quantiles
    #     ),  # QuantileRegression is set per default
    #     # loss_fn=MSELoss(),
    #     random_state=42,
    # )
    
    
    # my_model.fit(train_transformed, future_covariates=covariates_transformed, verbose=True)
    
    
    # testRes = my_model.predict(n=13, num_samples=200)
    
    
    # time = [pd.to_datetime('1958-01-01'), pd.to_datetime('1959-01-01')]
    # pred_tft_df = my_model.predict(n=13, num_samples=200).pd_dataframe().mean(axis=1)
    # pred_tft = pred_tft_df[time[0]: time[1]].values.reshape(-1,1)
    # y_true_tft = val_transformed.pd_dataframe()['#Passengers'][time[0]: time[1]].values.reshape(-1,1)
    # pred_tft_rescaled = StandardScaler().fit_transform(pred_tft)
    # y_true_tft_rescaled = StandardScaler().fit_transform(y_true_tft)


    # plt.plot(y_true_tft_rescaled)
    # plt.plot(pred_tft_rescaled)