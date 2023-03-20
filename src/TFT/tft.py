# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 08:36:05 2023

@author: othma
"""

import numpy as np

from os.path import abspath, dirname
import sys

from darts.timeseries import TimeSeries as tsFormat
from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression

from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

# import running folder: temporary fix
## directories path
directory = dirname(abspath(__file__))
runningDirectory = dirname(dirname(directory))
## add path
sys.path.append(runningDirectory)

from parameters import QUANTILES_TFT

import matplotlib.pyplot as plt

class AirTFT(object):
    
    def __init__(self, input_chunk_len, fcst_horizon,
                 hidden_size = 64, lstm_layers = 1, 
                 attention_heads = 4, dropout = 0.1, batch_size = 16,
                 epochs = 300, rnd_state = 42, nb_samples = 200):
        
        self.input_chunk_len = input_chunk_len
        self.fcst_horizon = fcst_horizon
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.attention_heads = attention_heads
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.rnd_state = rnd_state
        self.quantiles = QUANTILES_TFT
        self.nb_samples = nb_samples
        
        self.model = TFTModel(input_chunk_length = self.input_chunk_len,
                            output_chunk_length = self.fcst_horizon,
                            hidden_size = self.hidden_size,
                            lstm_layers = self.lstm_layers,
                            num_attention_heads = attention_heads,
                            dropout = self.dropout,
                            batch_size = self.batch_size,
                            n_epochs = self.epochs,
                            add_relative_index = False,
                            add_encoders = None,
                            likelihood=QuantileRegression(
                                quantiles=self.quantiles
                            ),  # QuantileRegression is set per default
                            # loss_fn=MSELoss(),
                            random_state=self.rnd_state
                            )
                    
    def fit(self, train: tsFormat, covariates: tsFormat) -> None:
        self.model.fit(train, future_covariates=covariates, verbose=True)
    

    def predict(self, horizon: int) -> np.ndarray:
        return self.model.predict(n=horizon, num_samples=200)


if __name__ == "__main__":
    # simple test of functions 
    ## import libraries
    from pre_process_data_tft import AirDataPreProcessorTFT, split_air_data_darts
    
    ## load data
    air_preprocessor = AirDataPreProcessorTFT()
    dataAir, covariates  = air_preprocessor.get_air_data()
    train, test = split_air_data_darts(dataAir)
    cov_train, cov_test = split_air_data_darts(covariates)

    ## build model and fit 
    input_chunk_len = 24
    fcst_horizon = 12
    modelTFT = AirTFT(input_chunk_len, fcst_horizon)
    modelTFT.fit(train, covariates)

    ## test/evaluate the model
    horizon = len(test)
    yTest = test.pd_dataframe()['#Passengers']
    predTest = modelTFT.predict(horizon).pd_dataframe().mean(axis=1)
    error = mean_squared_error(predTest, yTest)
    print(f" error: {error}")
    
    ## plot values 
    plt.plot(predTest)
    plt.plot(yTest)
    plt.show()