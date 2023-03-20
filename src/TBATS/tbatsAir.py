# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 08:36:05 2023

@author: othma
"""

from os.path import abspath, dirname
import sys

import pandas as pd
import numpy as np
from tbats import TBATS


# import running folder: temporary fix
## directories path
directory = dirname(abspath(__file__))
runningDirectory = dirname(dirname(directory))
## add path
sys.path.append(runningDirectory)

class AirTbats(object):
    
    def __init__(self, seasonal_periods: list):
        self.seasonal_periods = seasonal_periods 
        
        self.model = TBATS(seasonal_periods=self.seasonal_periods) 
        self.model_fitted = None
        
    def fit(self, yTrain: np.ndarray) -> None:
        self.model_fitted = self.model.fit(yTrain)

    def predict(self, horizon: int = 0) -> pd.Series:
        if self.model_fitted is not None:
            return self.model_fitted.forecast(steps=horizon)


if __name__ == "__main__":
    # simple test of functions 
    from src.data_pre_processing.preprocess_data import AirDataPreProcessor, split_air_data
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    
    ## load and preprocess data 
    air_preprocess = AirDataPreProcessor()
    dataAir = air_preprocess.get_air_data()
    
    ## split train/test
    train, test = split_air_data(dataAir)

    ## split features and targets
    yTrain = train['#Passengers'].values

    yTest = test['#Passengers'].values

    
    ## build model and fit 
    seasonal_periods=[12]
    modelTbats = AirTbats(seasonal_periods)
    modelTbats.fit(yTrain)
    
    ## test/evaluate the model   
    horizon = len(yTest)
    predTest = modelTbats.predict(horizon)
    error = mean_squared_error(predTest, yTest)
    print(f" error: {error}")
    
    ## plot values 
    plt.plot(predTest)
    plt.plot(yTest)
    plt.show()