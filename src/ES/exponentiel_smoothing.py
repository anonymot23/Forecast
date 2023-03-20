# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 08:36:05 2023

@author: othma
"""

from os.path import abspath, dirname
import sys

import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing

# import running folder: temporary fix
## directories path
directory = dirname(abspath(__file__))
runningDirectory = dirname(dirname(directory))
## add path
sys.path.append(runningDirectory)

class AirExpSmoothing(object):
    
    def __init__(self, lag: int = 12, trend: str = "add", seasonal: str = "add",
                 use_boxcox: bool = True):
        self.model = None 
        self.model_fitted = None
        self.min_val_train = 0
        
        self.lag = lag
        self.trend = trend 
        self.seasonal = seasonal
        self.use_boxcox = use_boxcox
    
    def initialize_model(self, yTrain: pd.Series) -> None:
        self.min_val_train = yTrain.min()-0.1
        self.model = ExponentialSmoothing(yTrain-self.min_val_train, 
                    seasonal_periods=self.lag, 
                    trend=self.trend,
                    seasonal=self.seasonal,
                    use_boxcox=self.use_boxcox,
                    initialization_method='estimated')
                    
    def fit(self, yTrain: pd.Series) -> None:
        if self.model is None:
            self.initialize_model(yTrain)

        self.model_fitted = self.model.fit()
    
    def predict(self, max_horizon: int = 1) -> pd.Series:
        if self.model_fitted is not None:
            return self.model_fitted.forecast(max_horizon) + self.min_val_train


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
    
    ## reformat data for model
    yTrain = pd.Series(train['#Passengers'])
    yTest = pd.Series(test['#Passengers'])
    
    ## build model and fit 
    modelES = AirExpSmoothing()
    modelES.fit(yTrain)
    
    ## test/evaluate the model  
    max_horizon = len(yTest) 
    predTest = modelES.predict(max_horizon)
    error = mean_squared_error(predTest, yTest)
    print(f" error: {error}")
    
    ## plot values 
    plt.plot(predTest)
    plt.plot(yTest)
    plt.show()