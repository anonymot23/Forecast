# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
    
from src.LGBM.lgbm import AirLgbm

class AirForecastGeneratorLgbm(object):
    
    def __init__(self, train: bool = True):
        self.train = train
    
    def get_forecast(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
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
        predTest : TYPE
            DESCRIPTION.

        """
        ## split features and targets
        xTrain = train[['before_shift', 'before_lag']].fillna(0).to_numpy()
        yTrain = train['#Passengers'].values
    
    
        xTest = test[['before_shift', 'before_lag']].fillna(0).to_numpy()


        # build model and fit 
        model = AirLgbm()
        if self.train:
            model.fit(yTrain, xTrain)
        
        ## generate prediction 
        predTest = model.predict(xTest)
        
        return predTest
    
    