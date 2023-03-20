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

from src.PROPHET.prophetAir import AirProphet


class AirForecastGeneratorProphet(object):
    
    def __init__(self, train: bool = True):
        self.train = train
    
    def get_forecast(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray: 
        ## split features and targets
        yTrain = train['#Passengers'].values
        dfTrain = pd.DataFrame.from_dict({"ds": train.index, "y": yTrain.flatten()})
    
        dfTest = pd.DataFrame.from_dict({"ds": test.index})
        xTest = dfTest[['ds']]
    
        # build model and fit 
        modelProphet = AirProphet()
        if self.train:
            modelProphet.fit(dfTrain)
        
        ## generate prediction 
        predTest = modelProphet.predict(xTest)
        
        return predTest
    
    