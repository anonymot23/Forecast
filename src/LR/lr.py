# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 08:36:05 2023

@author: othma
"""

from os.path import abspath, dirname
import sys

import numpy as np
from sklearn.linear_model import LinearRegression

# import running folder: temporary fix
## directories path
directory = dirname(abspath(__file__))
runningDirectory = dirname(dirname(directory))
## add path
sys.path.append(runningDirectory)

class AirLr(object):
    
    def __init__(self):
        self.model = LinearRegression() 
        self.model_fitted = None
                    
    def fit(self, yTrain: np.ndarray, xTrain: np.ndarray) -> None:
        self.model_fitted = self.model.fit(xTrain, yTrain)

    def predict(self, xTest: np.ndarray) -> np.ndarray:
        return self.model_fitted.predict(xTest)


if __name__ == "__main__":
    # simple test of functions 
    from pre_process_data_lr import AirDataPreProcessorLr
    from src.data_pre_processing.preprocess_data import split_air_data
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    
    ## load and preprocess data 
    shift = 1
    lag = 12
    air_preprocess = AirDataPreProcessorLr(shift, lag)
    dataAir = air_preprocess.get_air_data()
    
    ## split train/test
    train, test = split_air_data(dataAir)

    ## split features and targets
    xTrain = train[['before_shift', 'before_lag']].fillna(0).to_numpy()#.values
    yTrain = train['#Passengers'].values

    xTest = test[['before_shift', 'before_lag']].fillna(0).to_numpy()#.values
    yTest = test['#Passengers'].values

    
    ## build model and fit 
    modelLr = AirLr()
    modelLr.fit(yTrain, xTrain)
    
    ## test/evaluate the model   
    predTest = modelLr.predict(xTest)
    error = mean_squared_error(predTest, yTest)
    print(f" error: {error}")
    
    ## plot values 
    plt.plot(predTest)
    plt.plot(yTest)
    plt.show()