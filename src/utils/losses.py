# -*- coding: utf-8 -*-

import numpy as np


def SWAPE(y, y_pred): 
    num = np.absolute(y - y_pred).sum()
    denom = (np.absolute(y).sum() + np.absolute(y_pred).sum()) / 2
    return num / denom

def SMAPE(y, y_pred): 
    num = np.absolute(y - y_pred)
    denom = np.absolute(y) + np.absolute(y_pred) 
    ratio = num / denom
    return ratio.mean()
