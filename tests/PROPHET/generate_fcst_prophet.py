# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
    
from src.PROPHET.prophetAir import AirProphet

class AirForecastGeneratorProphet(object):
    
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
        # split features and targets
        yTrain = train['#Passengers'].values
        dfTrain = pd.DataFrame.from_dict({"ds": train.index, "y": yTrain.flatten()})
    
        dfTest = pd.DataFrame.from_dict({"ds": test.index})
        xTest = dfTest[['ds']]
    
        # build model and fit 
        model = AirProphet()
        if self.train:
            model.fit(dfTrain)
        
        ## generate prediction 
        predTest = model.predict(xTest)
        
        return predTest
    
    