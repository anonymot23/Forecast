# -*- coding: utf-8 -*-

import numpy as np
from darts.timeseries import TimeSeries as tsFormat

from src.TFT.tft import AirTFT


class AirForecastGeneratorTFT(object):
    
    def __init__(self, input_chunk_len: int = 24, fcst_horizon: int = 12,
                 train: bool = True):
        self.input_chunk_len = input_chunk_len
        self.fcst_horizon = fcst_horizon
        self.train = train
    
    def get_forecast(self, train: tsFormat, covariates: tsFormat,
                     test: tsFormat) -> np.ndarray:
        """
        Generate forecast

        Parameters
        ----------
        train : tsFormat
            DESCRIPTION.
        covariates : tsFormat
            DESCRIPTION.
        test : tsFormat
            DESCRIPTION.

        Returns
        -------
        predTest : TYPE
            DESCRIPTION.

        """

        ## build model and fit 
        model = AirTFT(self.input_chunk_len, self.fcst_horizon)
        model.fit(train, covariates)
    
        ## generate prediction 
        horizon = len(test)
        predTest = model.predict(horizon).pd_dataframe().mean(axis=1)

        return predTest
    
    