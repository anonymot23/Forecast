# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from prophet import Prophet

class AirProphet(object):
    
    def __init__(self, changepoint_prior_scale: float = 0.1, seasonality_prior_scale: int = 1):
        self.changepoint_prior_scale= changepoint_prior_scale 
        self.seasonality_prior_scale = seasonality_prior_scale
        
        
        self.model = Prophet(changepoint_prior_scale=self.changepoint_prior_scale,
                             seasonality_prior_scale=self.seasonality_prior_scale) 
                    
    def fit(self, dfTrain: pd.DataFrame) -> None:
        """
        Fit model's parameters

        Parameters
        ----------
        dfTrain : pd.DataFrame
            DESCRIPTION.

        Returns
        -------
        None
            DESCRIPTION.

        """
        self.model.fit(dfTrain)

    def predict(self, xTest: pd.DataFrame) -> np.ndarray:
        """
        Predict values

        Parameters
        ----------
        xTest : pd.DataFrame
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return self.model.predict(xTest)['yhat']


if __name__ == "__main__":
    # Test functions 
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