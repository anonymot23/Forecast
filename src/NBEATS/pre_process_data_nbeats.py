# -*- coding: utf-8 -*-

from typing import Tuple

import pandas as pd
import numpy as np

from darts.datasets import AirPassengersDataset
from darts.timeseries import TimeSeries as tsFormat
from darts import TimeSeries
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.dataprocessing.transformers import Scaler
from darts.utils.model_selection import train_test_split

class AirDataPreProcessorDarts(object):
    
    def __init__(self, shift: int = 1, lag: int = 1):
        self._shift = shift
        self._lag = lag 
        
        self._scaler = None
        self._scaler_covs = None
    
    @property
    def shift(self):
        return self._shift 

    @property
    def lag(self):
        return self._lag 

    @property
    def scaler(self):
        return self._scaler 

    @property
    def scaler_covs (self):
        return self._scaler_covs 
    
    @scaler.setter
    def scaler(self, new_scaler):
        if isinstance(new_scaler, Scaler):
            self._scaler = new_scaler
        else:
            raise TypeError("Only Scaler type is allowed")
            
    @scaler_covs.setter
    def scaler_covs(self, new_scaler):
        if isinstance(new_scaler, Scaler):
            self._scaler_covs = new_scaler
        else:
            raise TypeError("Only Scaler type is allowed")
            
    def get_air_data(self) -> Tuple[tsFormat, tsFormat]:
        """
        Load and pre-process air data

        Returns
        -------
        Tuple[tsFormat, tsFormat]
            DESCRIPTION.

        """
        # load air data
        dataAir = load_air_data_darts()
        
        # pre-process air data 
        series, covariates, self.scaler, self.scaler_covs = pre_process_data_darts(dataAir)
        
        return series, covariates
    
    def save_air_darts(self, data: pd.DataFrame) -> None:
        """
        Save air data. Not implemented yet

        Parameters
        ----------
        data : pd.DataFrame
            DESCRIPTION.

        Returns
        -------
        None
            DESCRIPTION.

        """
        pass    
    

def load_air_data_darts() -> pd.DataFrame:
    """
    Load air data

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return AirPassengersDataset().load()

def pre_process_data_darts(series: tsFormat) -> Tuple:
    """
    Pre-process air data

    Parameters
    ----------
    series : tsFormat
        DESCRIPTION.

    Returns
    -------
    Tuple
        DESCRIPTION.

    """
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


def split_air_data_darts(series: tsFormat, option: str = 'periods',
                   training_cutoff: pd.Timestamp("19571201") = None) -> Tuple[tsFormat, tsFormat]:
    """
    Test and train splitting of air data

    Parameters
    ----------
    series : tsFormat
        DESCRIPTION.
    option : str, optional
        DESCRIPTION. The default is 'periods'.
    training_cutoff : pd.Timestamp("19571201"), optional
        DESCRIPTION. The default is None.

    Returns
    -------
    Tuple[tsFormat, tsFormat]
        DESCRIPTION.

    """

    if option == "periods":
        if training_cutoff is None:
            training_cutoff = pd.Timestamp("19571201")
        
        train, test = series.split_after(training_cutoff)
    elif option == "random":
        train, test = train_test_split(series, test_size=0.2, random_state=1368)

    return train, test
    
def filename_air_processed_darts(suff: str = "") -> str:
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
    return f"AirPassengersProcessedDarts_{suff}.csv"
    

if __name__ == "__main__":
    # Test functions
    dataAir = load_air_data_darts()
    dataAir, covariates, _, _ = pre_process_data_darts(dataAir)
    train, test = split_air_data_darts(dataAir)
    
    air_preprocessor = AirDataPreProcessorDarts()
    dataAir2, _  = air_preprocessor.get_air_data()

    print(dataAir.pd_dataframe())
    print(train.pd_dataframe())
    print(test.pd_dataframe())
    print(dataAir2.pd_dataframe())