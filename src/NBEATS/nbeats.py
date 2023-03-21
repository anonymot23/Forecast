# -*- coding: utf-8 -*-
import numpy as np

from darts.timeseries import TimeSeries as tsFormat
from darts.models import NBEATSModel

import torch 
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

class AirNbeats(object):
    
    def __init__(self, input_chunk_len: int, fcst_horizon: int,
                 num_stacks: int = 3, num_blocks: int = 3, num_layers: int = 3,
                 layer_exp: int = 7, dropout: float = 0.1, batch_size: int = 128,
                 epochs: int = 300, rnd_state: int = 42, nb_samples: int = 200,
                 lr: float = 1e-3, max_samples_per_ts: int = 180, num_workers: int = 0):
    
        self.input_chunk_len = input_chunk_len
        self.fcst_horizon = fcst_horizon
        self.num_stacks = num_stacks
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.layer_exp = layer_exp
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.rnd_state = rnd_state
        
        self.early_stopper = EarlyStopping("val_loss", min_delta=0.0001, patience=2, verbose=True)
        self.callbacks = [self.early_stopper]
        self.pl_trainer_kwargs = {"callbacks": self.callbacks}
        
        self.max_samples_per_ts = max_samples_per_ts
        self.num_workers = num_workers
        
        self.model = NBEATSModel(input_chunk_length = self.input_chunk_len,
                            output_chunk_length = self.fcst_horizon,
                            num_stacks = self.num_stacks,
                            num_blocks = self.num_blocks,
                            num_layers = self.num_layers,
                            layer_widths = 2**self.layer_exp,
                            dropout = self.dropout,
                            n_epochs = self.epochs,
                            batch_size = self.batch_size,
                            add_encoders = None,
                            likelihood = None,
                            loss_fn = torch.nn.MSELoss(),
                            random_state = self.rnd_state,
                            pl_trainer_kwargs = self.pl_trainer_kwargs,
                            force_reset = True,
                            save_checkpoints = True
                            )
                    
    def fit(self, train: tsFormat, covariates: tsFormat) -> None:
        """
        Fit model's parameters

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
        self.model.fit(
                series = train,
                val_series = train,
                past_covariates = covariates,
                val_past_covariates = covariates,
                max_samples_per_ts = self.max_samples_per_ts,
                num_loader_workers = self.num_workers
                )
    

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
    from pre_process_data_nbeats import AirDataPreProcessorDarts, split_air_data_darts
    
    ## load data
    air_preprocessor = AirDataPreProcessorDarts()
    dataAir, covariates  = air_preprocessor.get_air_data()
    train, test = split_air_data_darts(dataAir)
    cov_train, cov_test = split_air_data_darts(covariates)

    ## build model and fit 
    input_chunk_len = 64
    fcst_horizon = 24
    modelNbeats = AirNbeats(input_chunk_len, fcst_horizon)
    modelNbeats.fit(train, covariates)

    ## test/evaluate the model
    horizon = len(test)
    yTest = test.pd_dataframe()['#Passengers']
    predTest = modelNbeats.predict(horizon).pd_dataframe().mean(axis=1)
    error = mean_squared_error(predTest, yTest)
    print(f" error: {error}")
    
    ## plot values 
    plt.plot(predTest)
    plt.plot(yTest)
    plt.show()