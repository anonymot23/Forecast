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

from src.TBATS.tbatsAir import AirTbats


class AirForecastGeneratorTbats(object):
    
    def __init__(self, seasonal_periods: list = [12], train: bool = True):
        self.seasonal_periods = seasonal_periods 
        
        self.train = train
    
    def get_forecast(self, train: pd.DataFrame, horizon: int = 1) -> pd.Series:
        ## reformat data for model
        yTrain = pd.Series(train['#Passengers'])
    
        # build model and fit 
        modelSarimax = AirTbats(self.seasonal_periods)
        if self.train:
            modelSarimax.fit(yTrain)
        
        ## generate prediction 
        predTest = modelSarimax.predict(horizon)
        
        return predTest
    
    