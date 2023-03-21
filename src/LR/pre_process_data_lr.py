# -*- coding: utf-8 -*-

from os.path import join

import pandas as pd

from src.data_pre_processing.preprocess_data import AirDataPreProcessor
from parameters import DATA_FOLDER

class AirDataPreProcessorLr(AirDataPreProcessor):
    
    def __init__(self, shift: int = 1, lag: int = 1):
        self._shift = shift
        self._lag = lag 
        
        self._scaler = None
    
    @property
    def shift(self):
        return self._shift 

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
        dataAir = self.preprocess_data(dataAir, self.shift, self.lag)
        
        return dataAir
    
    def preprocess_data(self, data: pd.DataFrame, 
                        shift: int = 1, lag: int = 1) -> pd.DataFrame:
        """
        Pre-process air data
        
        Parameters
        ----------
        data : pd.DataFrame
            DESCRIPTION.
        shift : int, optional
            DESCRIPTION. The default is 1.
        lag : int, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        data : pd.DataFrame
            DESCRIPTION.

        """
        ## prepare features
        data["before_shift"] = data["#Passengers"].shift(self.shift)
        data["before_lag"] = data["#Passengers"].shift(lag)
        
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
        filename_air_save = filename_air_processed_lr()
        filepath = join(DATA_FOLDER, filename_air_save)
        data.to_csv(filepath)        
        
        
def filename_air_processed_lr(suff: str = "") -> str:
    """
    Returns the name of the file used to save processed air data

    Parameters
    ----------
    suff : str, optional
        DESCRIPTION. The default is "".

    Returns
    -------
    str
        DESCRIPTION.

    """
    return f"AirPassengersProcessedLr_{suff}.csv"
    
 
if __name__ == "__main__":
    # Test functions
    shift = 1
    lag = 12
    air_preprocessor = AirDataPreProcessorLr(shift, lag)
    dataAir = air_preprocessor.get_air_data()
    air_preprocessor.save_air(dataAir)

    print(dataAir)