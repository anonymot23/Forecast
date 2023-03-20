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


class AirDataPreProcessorBaseline(AirDataPreProcessor):
    
    def __init__(self, window: int = 2, lag: int = 1):
        self.window = window
        self.lag = lag 
        
        self.scaler = None
    
    def get_air_data(self) -> None:
        # call parent pre-processing
        dataAir = super().get_air_data()
        
        # additional pre-processing
        dataAir = self.preprocess_data(dataAir, self.window, self.lag)
        
        return dataAir
    
    def preprocess_data(self, data: pd.DataFrame, 
                        windowSize: int = 2, lag: int = 1) -> pd.DataFrame:
        ## prepare features
        data['MA'] = data['#Passengers'].rolling(windowSize).mean()
        data['MALag'] = data['MA'].shift(lag)
        
        return data
    
    def save_air(self, data: pd.DataFrame) -> None:
        filename_air_save = filename_air_processed_baseline()
        filepath = join(DATA_FOLDER, filename_air_save)
        data.to_csv(filepath)        
        
        
# filename for saving processed data
def filename_air_processed_baseline(suff: str = "") -> str:
    return f"AirPassengersProcessedBaseline_{suff}.csv"
    
 
if __name__ == "__main__":
    # first test of functions
    air_preprocessor = AirDataPreProcessorBaseline()
    dataAir = air_preprocessor.get_air_data()
    air_preprocessor.save_air(dataAir)

    print(dataAir)