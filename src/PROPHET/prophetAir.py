# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 08:36:05 2023

@author: othma
"""

from os.path import abspath, dirname
import sys

import pandas as pd
import numpy as np
from prophet import Prophet

# import running folder: temporary fix
## directories path
directory = dirname(abspath(__file__))
runningDirectory = dirname(dirname(directory))
## add path
sys.path.append(runningDirectory)

class AirProphet(object):
    
    def __init__(self, changepoint_prior_scale: float = 0.1, seasonality_prior_scale: int = 1):
        self.changepoint_prior_scale= changepoint_prior_scale 
        self.seasonality_prior_scale = seasonality_prior_scale
        
        
        self.model = Prophet(changepoint_prior_scale=self.changepoint_prior_scale,
                             seasonality_prior_scale=self.seasonality_prior_scale) 
                    
    def fit(self, dfTrain: pd.DataFrame) -> None:
        self.model.fit(dfTrain)

    def predict(self, xTest: pd.DataFrame) -> np.ndarray:
        return self.model.predict(xTest)['yhat']


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
    dfTrain = pd.DataFrame.from_dict({"ds": train.index, "y": yTrain.flatten()})

    yTest = test['#Passengers'].values
    dfTest = pd.DataFrame.from_dict({"ds": test.index, "y": yTest.flatten()})
    xTest = dfTest[['ds']]
    
    ## build model and fit 
    changepoint_prior_scale = 0.1
    seasonality_prior_scale = 1
    modelProphet = AirProphet(changepoint_prior_scale, seasonality_prior_scale)
    modelProphet.fit(dfTrain)
    
    ## test/evaluate the model   
    predTest = modelProphet.predict(xTest)
    error = mean_squared_error(predTest, yTest)
    print(f" error: {error}")
    
    ## plot values 
    plt.plot(predTest)
    plt.plot(yTest)
    plt.show()