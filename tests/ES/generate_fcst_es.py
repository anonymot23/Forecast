# -*- coding: utf-8 -*-

import pandas as pd
    
from src.ES.exponentiel_smoothing import AirExpSmoothing

class AirForecastGenerator(object):
    
    def __init__(self, train: bool = True):
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
        model = AirExpSmoothing()
        if self.train:
            model.fit(yTrain)
        
        ## generate prediction 
        predTest = model.predict(horizon)
        
        return predTest
    
    