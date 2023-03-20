# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 09:36:44 2023

@author: othma
"""

from os.path import abspath, dirname, join
import sys

import pandas as pd
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
    
# import running folder: temporary fix
## directories path
directory = dirname(abspath(__file__))
runningDirectory = dirname(dirname(directory))
testDirectory = join(runningDirectory, "tests\ES")
## add path
sys.path.append(runningDirectory)
sys.path.append(testDirectory)

from src.data_pre_processing.preprocess_data import AirDataPreProcessor, split_air_data
from generate_fcst_sarimax import AirForecastGeneratorSarimax


def test_air_model(yTest: pd.Series, predTest: pd.Series) -> dict():
    # compute error
    error = mean_squared_error(predTest, yTest)
    
    return {"error": error}

        
def main_air_sarimax(params: dict = {}) -> None:
    # load and preprocess data 
    air_preprocess = AirDataPreProcessor()
    dataAir = air_preprocess.get_air_data()
    
    # split train/test
    train, test = split_air_data(dataAir)
    
    # generate forecast
    max_horizon = len(test) 
    air_fcst_generator = AirForecastGeneratorSarimax(train=True)
    predTest = air_fcst_generator.get_forecast(train, max_horizon)
    
    # test/evaluate the model  
    targetTest = test['#Passengers']
    dictError = test_air_model(targetTest, predTest)
    print(f""" error: {dictError["error"]}""")
    
    # plot values 
    plt.plot(predTest)
    plt.plot(targetTest)
    plt.show()
    
    
if __name__ == "__main__":
    # simple test of functions 
    params = dict()
    main_air_sarimax(params)
    
    