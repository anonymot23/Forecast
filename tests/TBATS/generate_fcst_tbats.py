# -*- coding: utf-8 -*-

import pandas as pd

from src.TBATS.tbatsAir import AirTbats


class AirForecastGeneratorTbats(object):
    
    def __init__(self, seasonal_periods: list = [12], train: bool = True):
        self.seasonal_periods = seasonal_periods 
        
        self.train = train
    
    def get_forecast(self, train: pd.DataFrame, horizon: int = 1) -> pd.Series:
        """
        Generate forecast

        Parameters
        ----------
        train : pd.DataFrame
            DESCRIPTION.
        horizon : int, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        predTest : pd.Series
            DESCRIPTION.

        """
        ## reformat data for model
        yTrain = pd.Series(train['#Passengers'])
    
        # build model and fit 
        model = AirTbats(self.seasonal_periods)
        if self.train:
            model.fit(yTrain)
        
        ## generate prediction 
        predTest = model.predict(horizon)
        
        return predTest
    
    