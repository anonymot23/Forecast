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

from src.BASELINE.baseline import AirBaseline


class AirForecastGeneratorBaseline(object):
    
    def __init__(self, train: bool = True):
        self.train = train
    
    def get_forecast(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        ## remove NA from train
        filterNA = train['MALag'] == train['MALag']
        train = train[filterNA]
    
        ## split features and targets
        xTrain = train[['MALag']].values
        yTrain = train['#Passengers'].values
        
        xTest = test[['MALag']].values
    
        # build model and fit 
        modelLR = AirBaseline()
        if self.train:
            modelLR.fit(yTrain, xTrain)
        
        ## generate prediction 
        predTest = modelLR.predict(xTest)
        
        return predTest
    
    