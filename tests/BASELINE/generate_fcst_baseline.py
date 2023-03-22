# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from src.BASELINE.baseline import AirBaseline

class AirForecastGeneratorBaseline(object):
    
    def __init__(self, train: bool = True):
        self.train = train
    
    def get_forecast(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        """
        Generate forecasted values

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
        # remove NA from train
        filterNA = train['MALag'] == train['MALag']
        train = train[filterNA]
    
        # split features and targets
        xTrain = train[['MALag']].values
        yTrain = train['#Passengers'].values
        
        xTest = test[['MALag']].values
    
        # build model and fit 
        model = AirBaseline()
        if self.train:
            model.fit(yTrain, xTrain)
        
        ## generate prediction 
        predTest = model.predict(xTest)
        
        return predTest
    
    