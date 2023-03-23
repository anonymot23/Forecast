# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from src.utils.losses import SMAPE, SWAPE


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
    error_mse = mean_squared_error(predTest, yTest)
    error_rmse = mean_squared_error(predTest, yTest, squared = False)
    error_mae = mean_absolute_error(predTest, yTest)
    error_mape = mean_absolute_percentage_error(predTest, yTest)
    error_smape = SMAPE(predTest, yTest)
    error_swape = SWAPE(predTest, yTest)
    
    error = {"mse": error_mse, "rmse": error_rmse, 
             "mae": error_mae, "mape": error_mape, 
             "smape": error_smape, "swape": error_swape}
    
    list1 = ["mse", "rmse", "mae", "mape", "smape", "swape"]
    print(" | ".join(["{:0.2f}".format(error[k]) for k in list1]))
    return {"error": error}
        