# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 09:11:15 2023

@author: othma
"""

from os.path import abspath, dirname
import sys

import numpy as np
from darts.timeseries import TimeSeries as tsFormat
    
# import running folder: temporary fix
## directories path
directory = dirname(abspath(__file__))
runningDirectory = dirname(dirname(directory))
## add path
sys.path.append(runningDirectory)

from src.NBEATS.nbeats import AirNbeats


class AirForecastGeneratorNbeats(object):
    
    def __init__(self, input_chunk_len: int = 64, fcst_horizon: int = 24,
                 train: bool = True):
        self.input_chunk_len = input_chunk_len
        self.fcst_horizon = fcst_horizon
        self.train = train
    
    def get_forecast(self, train: tsFormat, covariates: tsFormat,
                     test: tsFormat) -> np.ndarray: 

        ## build model and fit 
        modelNbeats = AirNbeats(self.input_chunk_len, self.fcst_horizon)
        modelNbeats.fit(train, covariates)
    
        ## generate prediction 
        horizon = len(test)
        predTest = modelNbeats.predict(horizon).pd_dataframe().mean(axis=1)

        return predTest
    
    