# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 09:54:56 2023

@author: othma
"""


from os.path import abspath, dirname
import sys
   
# import running folder: temporary fix
## directories path
directory = dirname(abspath(__file__))
runningDirectory = dirname(dirname(directory))
## add path
sys.path.append(runningDirectory)

from main_xgboost import main_air_xgboost
    
    
if __name__ == "__main__":
    # simple test of functions : add more tests
    params = dict()
    main_air_xgboost(params)
    
    