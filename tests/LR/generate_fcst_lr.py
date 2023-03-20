# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 09:11:15 2023

@author: othma
"""

from os.path import abspath, dirname
import sys

import pandas as pd
import numpy as np
    
# import running folder: temporary fix
## directories path
directory = dirname(abspath(__file__))
runningDirectory = dirname(dirname(directory))
## add path
sys.path.append(runningDirectory)

from src.LR.lr import AirLr


class AirForecastGeneratorLr(object):
    
    def __init__(self, train: bool = True):
        self.train = train
    
    def get_forecast(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        ## split features and targets
        xTrain = train[['before_shift', 'before_lag']].fillna(0).to_numpy()
        yTrain = train['#Passengers'].values
    
        xTest = test[['before_shift', 'before_lag']].fillna(0).to_numpy()

    
        # build model and fit 
        modelLr = AirLr()
        if self.train:
            modelLr.fit(yTrain, xTrain)
        
        ## generate prediction 
        predTest = modelLr.predict(xTest)
        
        return predTest
    
    