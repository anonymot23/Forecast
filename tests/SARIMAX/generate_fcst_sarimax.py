# -*- coding: utf-8 -*-

import pandas as pd

from src.SARIMAX.sarimax import AirSarimax


class AirForecastGeneratorSarimax(object):
    
    def __init__(self, order: tuple = (1, 0, 1), seasonal_order: tuple=(1, 0, 1, 12), train: bool = True):
        self.order = order
        self.seasonal_order = seasonal_order 
        
        self.train = train
    
    def get_forecast(self, train: pd.DataFrame, horizon: int = 1) -> pd.Series:
        """
        Generate forecast

        Parameters
        ----------
        train : pd.DataFrame
            DESCRIPTION.
        test : pd.DataFrame
            DESCRIPTION.

        Returns
        -------
        predTest : np.ndarray
            DESCRIPTION.

        """
        ## reformat data for model
        yTrain = pd.Series(train['#Passengers'])
    
        # build model and fit 
        modelSarimax = AirSarimax(self.order, self.seasonal_order)
        if self.train:
            modelSarimax.fit(yTrain)
        
        ## generate prediction 
        start = len(yTrain)
        end = start + horizon- 1
        predTest = modelSarimax.predict(start, end)
        
        return predTest
    
    