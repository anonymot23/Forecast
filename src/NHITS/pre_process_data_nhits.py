# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 00:27:39 2023

@author: othma
"""


from os.path import abspath, dirname
import sys

from typing import Tuple

import pandas as pd
import numpy as np

from darts.datasets import AirPassengersDataset
from darts.timeseries import TimeSeries as tsFormat
from darts import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.dataprocessing.transformers import Scaler
from darts.utils.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

# import running folder: temporary fix
## directories path
directory = dirname(abspath(__file__))
runningDirectory = dirname(dirname(directory))
## add path
sys.path.append(runningDirectory)


class AirDataPreProcessorNhits(object):
    
    def __init__(self, shift: int = 1, lag: int = 1):
        self.shift = shift
        self.lag = lag 
        
        self.scaler = None
        self.scaler_covs = None
    
    def get_air_data(self) -> None:
        # load air data
        dataAir = load_air_data_darts()
        
        # pre-process air data 
        series, covariates, self.scaler, self.scaler_covs = pre_process_data_darts(dataAir)
        
        return series, covariates
    
    def save_air_darts(self, data: pd.DataFrame) -> None:
        pass    
        
# load data 
def load_air_data_darts() -> pd.DataFrame:
    return AirPassengersDataset().load()

# pre-process data
def pre_process_data_darts(series: tsFormat) -> Tuple:
    # Convert monthly number of passengers to average daily number of passengers per month
    series = series / TimeSeries.from_series(series.time_index.days_in_month)
    series = series.astype(np.float32)
    
    
    # Normalize the time series (note: we avoid fitting the transformer on the validation set)
    transformer = Scaler()
    series = transformer.fit_transform(series)
    

    # create year, month and integer index covariate series
    covariates = datetime_attribute_timeseries(series, attribute="year", one_hot=False)
    covariates = covariates.stack(
        datetime_attribute_timeseries(series, attribute="month", one_hot=False)
    )
    covariates = covariates.stack(
        TimeSeries.from_times_and_values(
            times=series.time_index,
            values=np.arange(len(series)),
            columns=["linear_increase"],
        )
    )
    covariates = covariates.astype(np.float32)
    
        
    # transform covariates (note: we fit the transformer on train split and can then transform the entire covariates series)
    scaler_covs = Scaler()
    scaler_covs.fit(covariates)
    covariates = scaler_covs.transform(covariates)
        
    return series, covariates, transformer, scaler_covs


# split data
def split_air_data_darts(series: tsFormat, option: str = 'periods',
                   training_cutoff: pd.Timestamp("19571201") = None) -> Tuple[tsFormat, tsFormat]:

    if option == "periods":
        if training_cutoff is None:
            training_cutoff = pd.Timestamp("19571201")
        
        train, test = series.split_after(training_cutoff)
    elif option == "random":
        train, test = train_test_split(series, test_size=0.2, random_state=1368)

    return train, test
    
# filename for saving processed data
def filename_air_processed_xgboost(suff: str = "") -> str:
    return f"AirPassengersProcessedDarts_{suff}.csv"
    

if __name__ == "__main__":
    # first test of functions
    dataAir = load_air_data_darts()
    dataAir, covariates, _, _ = pre_process_data_darts(dataAir)
    train, test = split_air_data_darts(dataAir)
    
    air_preprocessor = AirDataPreProcessorNhits()
    dataAir2, _  = air_preprocessor.get_air_data()

    print(dataAir.pd_dataframe())
    print(train.pd_dataframe())
    print(test.pd_dataframe())
    print(dataAir2.pd_dataframe())