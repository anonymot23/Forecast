# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 00:27:39 2023

@author: othma
"""

from os.path import join, abspath, dirname
import sys

import pandas as pd



# import running folder: temporary fix
## directories path
directory = dirname(abspath(__file__))
runningDirectory = dirname(dirname(directory))
## add path
sys.path.append(runningDirectory)

from src.data_pre_processing.preprocess_data import AirDataPreProcessor
from parameters import DATA_FOLDER


class AirDataPreProcessorLgbm(AirDataPreProcessor):
    
    def __init__(self, shift: int = 1, lag: int = 1):
        self.shift = shift
        self.lag = lag 
        
        self.scaler = None
    
    def get_air_data(self) -> None:
        # call parent pre-processing
        dataAir = super().get_air_data()
        
        # additional pre-processing
        dataAir = self.preprocess_data(dataAir, self.shift, self.lag)
        
        return dataAir
    
    def preprocess_data(self, data: pd.DataFrame, 
                        shift: int = 1, lag: int = 1) -> pd.DataFrame:
        ## prepare features
        data["before_shift"] = data["#Passengers"].shift(self.shift)
        data["before_lag"] = data["#Passengers"].shift(lag)
        
        return data
    
    def save_air(self, data: pd.DataFrame) -> None:
        filename_air_save = filename_air_processed_lgbm()
        filepath = join(DATA_FOLDER, filename_air_save)
        data.to_csv(filepath)        
        
        
# filename for saving processed data
def filename_air_processed_lgbm(suff: str = "") -> str:
    return f"AirPassengersProcessedLgbm_{suff}.csv"
    
 
if __name__ == "__main__":
    # first test of functions
    shift = 1
    lag = 12
    air_preprocessor = AirDataPreProcessorLgbm(shift, lag)
    dataAir = air_preprocessor.get_air_data()
    air_preprocessor.save_air(dataAir)

    print(dataAir)