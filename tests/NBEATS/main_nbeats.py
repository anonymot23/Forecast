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


from src.NBEATS.pre_process_data_nbeats import AirDataPreProcessorNbeats, split_air_data_darts
from generate_fcst_nbeats import AirForecastGeneratorNbeats


def test_air_model(yTest: pd.Series, predTest: pd.Series) -> dict():
    # compute error
    error = mean_squared_error(predTest, yTest)
    
    return {"error": error}

        
def main_air_nbeats(params: dict = {}) -> None:
    # load and preprocess data 
    air_preprocessor = AirDataPreProcessorNbeats()
    dataAir, covariates  = air_preprocessor.get_air_data()
    
    # split train/test
    train, test = split_air_data_darts(dataAir)
    cov_train, cov_test = split_air_data_darts(covariates)
    
    # generate forecast
    air_fcst_generator = AirForecastGeneratorNbeats(train=True)
    predTest = air_fcst_generator.get_forecast(train, covariates, test)
    
    # test/evaluate the model  
    yTest = test.pd_dataframe()['#Passengers']
    dictError = test_air_model(yTest, predTest)
    print(f""" error: {dictError["error"]}""")
    
    # plot values 
    plt.plot(predTest)
    plt.plot(yTest)
    plt.show()
    
    
if __name__ == "__main__":
    # simple test of functions 
    params = dict()
    main_air_nbeats(params)
    
    