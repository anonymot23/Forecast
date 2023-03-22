# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

from src.data_pre_processing.preprocess_data import split_air_data
from src.LR.pre_process_data_lr import AirDataPreProcessorLr
from generate_fcst_lr import AirForecastGeneratorLr


def test_air_model(yTest: pd.Series, predTest: pd.Series) -> dict():
    """
    Compute test metrics

    Parameters
    ----------
    yTest : pd.Series
        DESCRIPTION.
    predTest : pd.Series
        DESCRIPTION.

    Returns
    -------
    dict()
        DESCRIPTION.

    """
    # compute error
    error = mean_squared_error(predTest, yTest)
    
    return {"error": error}

        
def main_air_lr(params: dict = {}) -> None:
    """
    Model's runner

    Parameters
    ----------
    params : dict, optional
        DESCRIPTION. The default is {}.

    Returns
    -------
    None
        DESCRIPTION.

    """
    # load and preprocess data 
    shift = 1
    lag = 12
    air_preprocess = AirDataPreProcessorLr(shift, lag)
    dataAir = air_preprocess.get_air_data()
    
    # split train/test
    train, test = split_air_data(dataAir)
    
    # generate forecast
    air_fcst_generator = AirForecastGeneratorLr(train=True)
    predTest = air_fcst_generator.get_forecast(train, test)
    
    # test/evaluate the model  
    yTest = test['#Passengers'].values
    dictError = test_air_model(yTest, predTest)
    print(f""" error: {dictError["error"]}""")
    
    # plot values 
    plt.plot(predTest)
    plt.plot(yTest)
    plt.show()
    
    
if __name__ == "__main__":
    # Test functions
    params = dict()
    main_air_lr(params)
    
    