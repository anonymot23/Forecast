# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 18:41:39 2023

@author: othma
"""

from typing import Tuple
from os.path import join, abspath, dirname
import sys

import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# import running folder: temporary fix
## directories path
directory = dirname(abspath(__file__))
runningDirectory = dirname(dirname(directory))
## add path
sys.path.append(runningDirectory)

from parameters import DATA_FOLDER, AIR_PASSENGERS_FILENAME


class AirDataPreProcessor(object):
    
    def __init__(self):
        self.scaler = None
    
    def get_air_data(self) -> None:
        # load air data
        dataAir = load_air_data()
        
        # pre-process air data 
        dataAir, self.scaler = pre_process_data(dataAir)
        
        return dataAir
    
    def save_air(self, data: pd.DataFrame) -> None:
        filename_air_save = filename_air_processed()
        filepath = join(DATA_FOLDER, filename_air_save)
        data.to_csv(filepath)        
        
        
    
# load data 
def load_air_data() -> pd.DataFrame:
    filepath = join(DATA_FOLDER, AIR_PASSENGERS_FILENAME)
    return pd.read_csv(filepath)


# pre-process data
def pre_process_data(dataAir: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    dataAir['Month'] = pd.to_datetime(dataAir['Month'])
    dataAir.set_index(dataAir['Month'], inplace=True)
    scaler = StandardScaler()
    passengers = dataAir['#Passengers'].values.reshape((-1, 1))
    scaler.fit(passengers)
    dataAir['#Passengers'] = scaler.transform(passengers)
    return dataAir, scaler

# split data
def split_air_data(data: pd.DataFrame, option: str = 'periods', random_state: int = 1368,
                   timeTrain: Tuple[datetime, datetime] = (None, None),
                   timeTest: Tuple[datetime, datetime] = (None, None)) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if option == "periods":
        if timeTrain[0] is None:
            timeTrain = [pd.to_datetime('1950-01-01') , pd.to_datetime('1957-12-01')]
        if timeTest[0] is None:
            timeTest = [pd.to_datetime('1958-01-01') , pd.to_datetime('1959-01-01')]
        
        train = data[timeTrain[0]: timeTrain[1]]
        # train = pd.Series(dataTrain['#Passengers'])
        
        test = data[timeTest[0]: timeTest[1]]
        # test = pd.Series(dataTest['#Passengers'])
    elif option == "random":
        train, test = train_test_split(data, test_size=0.2, random_state=1368)

    return train, test

# filename for saving processed data
def filename_air_processed(suff: str = "") -> str:
    return f"AirPassengersProcessed_{suff}.csv"
    
 
if __name__ == "__main__":
    # first test of functions
    dataAir = load_air_data()
    dataAir, _ = pre_process_data(dataAir)
    train, test = split_air_data(dataAir)
    
    air_preprocessor = AirDataPreProcessor()
    dataAir2 = air_preprocessor.get_air_data()

    print(dataAir)
    print(train)
    print(test)
    print(dataAir2)