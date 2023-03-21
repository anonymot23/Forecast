# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

class AirSarimax(object):
    
    def __init__(self, order: tuple, seasonal_order: tuple):
        self.model = None 
        self.model_fitted = None
        
        self.order = order
        self.seasonal_order = seasonal_order 

    def initialize_model(self, yTrain: pd.Series) -> None:
        """
        Initialize model parameters

        Parameters
        ----------
        yTrain : pd.Series
            DESCRIPTION.

        Returns
        -------
        None
            DESCRIPTION.

        """
        self.model = SARIMAX(yTrain,
                             order = self.order,
                             seasonal_order = self.seasonal_order )
        
    def fit(self, yTrain: np.ndarray) -> None:
        """
        Fit model's parameters

        Parameters
        ----------
        yTrain : np.ndarray
            DESCRIPTION.

        Returns
        -------
        None
            DESCRIPTION.

        """
        if self.model is None:
            self.initialize_model(yTrain)

        self.model_fitted = self.model.fit()

    def predict(self, start: int = 0, end: int = 1) -> pd.Series:
        """
        Predict values

        Parameters
        ----------
        start : int, optional
            DESCRIPTION. The default is 0.
        end : int, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if self.model_fitted is not None:
            return self.model_fitted.predict(start=start, end=end)


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

    yTest = test['#Passengers'].values

    
    ## build model and fit 
    order=(1, 0, 1)
    seasonal_order=(1, 0, 1, 12)
    modelSarimax = AirSarimax( order, seasonal_order)
    modelSarimax.fit(yTrain)
    
    ## test/evaluate the model   
    start = len(yTrain)
    end = start + len(yTest) - 1
    predTest = modelSarimax.predict(start=start, end=end)
    error = mean_squared_error(predTest, yTest)
    print(f" error: {error}")
    
    ## plot values 
    plt.plot(predTest)
    plt.plot(yTest)
    plt.show()