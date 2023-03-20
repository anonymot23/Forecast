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

class AirBaseline(object):
    
    def __init__(self):
        self.model = LinearRegression() 
                    
    def fit(self, yTrain: np.ndarray, xTrain: np.ndarray) -> None:
        self.model.fit(xTrain, yTrain)

    def predict(self, xTest: np.ndarray) -> np.ndarray:
        return self.model.predict(xTest)


if __name__ == "__main__":
    # simple test of functions 
    from pre_process_data_baseline import AirDataPreProcessorBaseline
    from src.data_pre_processing.preprocess_data import split_air_data
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    
    ## load and preprocess data 
    air_preprocess = AirDataPreProcessorBaseline()
    dataAir = air_preprocess.get_air_data()
    
    ## split train/test
    train, test = split_air_data(dataAir)
    
    ## remove NA from train
    filterNA = train['MALag'] == train['MALag']
    train = train[filterNA]

    ## split features and targets
    xTrain = train[['MALag']].values
    yTrain = train['#Passengers'].values
    
    xTest = test[['MALag']].values
    yTest = test['#Passengers'].values
    
    ## build model and fit 
    modelLR = AirBaseline()
    modelLR.fit(yTrain, xTrain)
    
    ## test/evaluate the model   
    predTest = modelLR.predict(xTest)
    error = mean_squared_error(predTest, yTest)
    print(f" error: {error}")
    
    ## plot values 
    plt.plot(predTest)
    plt.plot(yTest)
    plt.show()