# -*- coding: utf-8 -*-

from typing import Tuple
from os.path import join

import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from parameters import DATA_FOLDER, AIR_PASSENGERS_FILENAME


class AirDataPreProcessor(object):
    
    def __init__(self):
        self._scaler = None
    
    @property
    def scaler(self):
        return self._scaler 
    
    @scaler.setter
    def scaler(self, new_scaler):
        if isinstance(new_scaler, StandardScaler):
            self._scaler = new_scaler
        else:
            raise TypeError("Only StandardScaler type is allowed")
            
    def get_air_data(self) -> pd.DataFrame:
        """
        Load and pre-process air data

        Returns
        -------
        dataAir : pd.DataFrame
            DESCRIPTION.

        """
        # load air data
        dataAir = load_air_data()
        
        # pre-process air data 
        dataAir, self.scaler = pre_process_data(dataAir)
        
        return dataAir
    
    def save_air(self, data: pd.DataFrame) -> None:
        """
        Save air data

        Parameters
        ----------
        data : pd.DataFrame
            DESCRIPTION.

        Returns
        -------
        None
            DESCRIPTION.

        """
        filename_air_save = filename_air_processed()
        filepath = join(DATA_FOLDER, filename_air_save)
        data.to_csv(filepath)        
        
        
def load_air_data() -> pd.DataFrame:
    """
    Load air data

    Returns
    -------
    pd.DataFrame
        DESCRIPTION.

    """
    filepath = join(DATA_FOLDER, AIR_PASSENGERS_FILENAME)
    return pd.read_csv(filepath)

def pre_process_data(dataAir: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Pre-process air data

    Parameters
    ----------
    dataAir : pd.DataFrame
        DESCRIPTION.

    Returns
    -------
    dataAir : pd.DataFrame
        DESCRIPTION.
    scaler : StandardScaler
        DESCRIPTION.

    """
    dataAir['Month'] = pd.to_datetime(dataAir['Month'])
    dataAir.set_index(dataAir['Month'], inplace=True)
    scaler = StandardScaler()
    passengers = dataAir['#Passengers'].values.reshape((-1, 1))
    scaler.fit(passengers)
    dataAir['#Passengers'] = scaler.transform(passengers)
    return dataAir, scaler

def split_air_data(data: pd.DataFrame, option: str = 'periods', random_state: int = 1368,
                   timeTrain: Tuple[datetime, datetime] = (None, None),
                   timeTest: Tuple[datetime, datetime] = (None, None)) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Test and train splitting of air data

    Parameters
    ----------
    data : pd.DataFrame
        DESCRIPTION.
    option : str, optional
        DESCRIPTION. The default is 'periods'.
    random_state : int, optional
        DESCRIPTION. The default is 1368.
    timeTrain : Tuple[datetime, datetime], optional
        DESCRIPTION. The default is (None, None).
    timeTest : Tuple[datetime, datetime], optional
        DESCRIPTION. The default is (None, None).

    Returns
    -------
    train : pd.DataFrame
        DESCRIPTION.
    test : pd.DataFrame
        DESCRIPTION.

    """

    if option == "periods":
        if timeTrain[0] is None:
            timeTrain = [pd.to_datetime('1950-01-01') , pd.to_datetime('1957-12-01')]
        if timeTest[0] is None:
            timeTest = [pd.to_datetime('1958-01-01') , pd.to_datetime('1959-01-01')]
        
        train = data[timeTrain[0]: timeTrain[1]]
        
        test = data[timeTest[0]: timeTest[1]]
        
    elif option == "random":
        train, test = train_test_split(data, test_size=0.2, random_state=1368)

    return train, test

def filename_air_processed(suff: str = "") -> str:
    """
    Returns the name of the file used to save air data

    Parameters
    ----------
    suff : str, optional
        DESCRIPTION. The default is "".

    Returns
    -------
    str
        DESCRIPTION.

    """
    return f"AirPassengersProcessed_{suff}.csv"
    
 
if __name__ == "__main__":
    # Test functions
    dataAir = load_air_data()
    dataAir, _ = pre_process_data(dataAir)
    train, test = split_air_data(dataAir)
    
    air_preprocessor = AirDataPreProcessor()
    dataAir2 = air_preprocessor.get_air_data()

    print(dataAir)
    print(train)
    print(test)
    print(dataAir2)