# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 09:11:15 2023

@author: othma
"""

from os.path import abspath, dirname
import sys

import pandas as pd
    
# import running folder: temporary fix
## directories path
directory = dirname(abspath(__file__))
runningDirectory = dirname(dirname(directory))
## add path
sys.path.append(runningDirectory)

from src.SARIMAX.sarimax import AirSarimax


class AirForecastGeneratorSarimax(object):
    
    def __init__(self, order: tuple = (1, 0, 1), seasonal_order: tuple=(1, 0, 1, 12), train: bool = True):
        self.order = order
        self.seasonal_order = seasonal_order 
        
        self.train = train
    
    def get_forecast(self, train: pd.DataFrame, horizon: int = 1) -> pd.Series:
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
    
    