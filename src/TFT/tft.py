# -*- coding: utf-8 -*-
import numpy as np

from darts.timeseries import TimeSeries as tsFormat
from darts.models import TFTModel
from darts.utils.likelihood_models import QuantileRegression

from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

from parameters import QUANTILES_TFT

import matplotlib.pyplot as plt

class AirTFT(object):
    
    def __init__(self, input_chunk_len: int, fcst_horizon: int,
                 hidden_size: int = 64, lstm_layers: int = 1, 
                 attention_heads: int = 4, dropout: float = 0.1,
                 batch_size: int = 16, epochs: int = 300,
                 rnd_state: int = 42, nb_samples: int = 200):
        
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
        """
        Fit models's parameters

        Parameters
        ----------
        train : tsFormat
            DESCRIPTION.
        covariates : tsFormat
            DESCRIPTION.

        Returns
        -------
        None
            DESCRIPTION.

        """
        self.model.fit(train, future_covariates=covariates, verbose=True)
    

    def predict(self, horizon: int) -> np.ndarray:
        """
        Predict values

        Parameters
        ----------
        horizon : int
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.model.predict(n=horizon, num_samples=200)


if __name__ == "__main__":
    # Test functions 
    ## import libraries
    from src.NBEATS.pre_process_data_nbeats import AirDataPreProcessorDarts, split_air_data_darts
    
    ## load data
    air_preprocessor = AirDataPreProcessorDarts()
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