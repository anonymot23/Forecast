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

from src.ES.exponentiel_smoothing import AirExpSmoothing


class AirForecastGenerator(object):
    
    def __init__(self, train: bool = True):
        self.train = train
    
    def get_forecast(self, train: pd.DataFrame, horizon: int = 1) -> pd.Series:
        ## reformat data for model
        yTrain = pd.Series(train['#Passengers'])
    
        # build model and fit 
        modelES = AirExpSmoothing()
        if self.train:
            modelES.fit(yTrain)
        
        ## generate prediction 
        predTest = modelES.predict(horizon)
        
        return predTest
    
    