# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

from src.TFT.pre_process_data_tft import AirDataPreProcessorTFT, split_air_data_darts
from generate_fcst_tft import AirForecastGeneratorTFT


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

        
def main_air_tft(params: dict = {}) -> None:
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
    air_preprocessor = AirDataPreProcessorTFT()
    dataAir, covariates  = air_preprocessor.get_air_data()
    
    # split train/test
    train, test = split_air_data_darts(dataAir)
    cov_train, cov_test = split_air_data_darts(covariates)
    
    # generate forecast
    air_fcst_generator = AirForecastGeneratorTFT(train=True)
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
    # Test functions 
    params = dict()
    main_air_tft(params)
    
    