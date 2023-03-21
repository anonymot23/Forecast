# -*- coding: utf-8 -*-
from os.path import join

import pandas as pd

from src.data_pre_processing.preprocess_data import AirDataPreProcessor
from parameters import DATA_FOLDER


class AirDataPreProcessorBaseline(AirDataPreProcessor):
    
    def __init__(self, window: int = 2, lag: int = 1):
        self._window = window
        self._lag = lag 
        
        self._scaler = None
    
    @property
    def window(self):
        return self._window 

    @property
    def lag(self):
        return self._lag 
    
    def get_air_data(self) -> pd.DataFrame:
        """
        Load and pre-process air data

        Returns
        -------
        dataAir : pd.DataFrame
            DESCRIPTION.

        """
        # call parent pre-processing
        dataAir = super().get_air_data()
        
        # additional pre-processing
        dataAir = self.preprocess_data(dataAir, self.window, self.lag)
        
        return dataAir
    
    def preprocess_data(self, data: pd.DataFrame, 
                        windowSize: int = 2, lag: int = 1) -> pd.DataFrame:
        """
        Pre-process air data

        Parameters
        ----------
        data : pd.DataFrame
            DESCRIPTION.
        windowSize : int, optional
            DESCRIPTION. The default is 2.
        lag : int, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        data : pd.DataFrame
            DESCRIPTION.

        """
        # prepare features
        data['MA'] = data['#Passengers'].rolling(windowSize).mean()
        data['MALag'] = data['MA'].shift(lag)
        
        return data
    
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
        filename_air_save = filename_air_processed_baseline()
        filepath = join(DATA_FOLDER, filename_air_save)
        data.to_csv(filepath)        
        
        
def filename_air_processed_baseline(suff: str = "") -> str:
    """
    Returns the name of the file use to save air data

    Parameters
    ----------
    suff : str, optional
        DESCRIPTION. The default is "".

    Returns
    -------
    str
        DESCRIPTION.

    """
    return f"AirPassengersProcessedBaseline_{suff}.csv"
    

if __name__ == "__main__":
    # Test functions
    air_preprocessor = AirDataPreProcessorBaseline()
    dataAir = air_preprocessor.get_air_data()
    air_preprocessor.save_air(dataAir)

    print(dataAir)