# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 09:54:56 2023

@author: othma
"""


from os.path import abspath, dirname, join
import sys
   
# import running folder: temporary fix
## directories path
directory = dirname(abspath(__file__))
runningDirectory = dirname(dirname(directory))
testDirectory = join(runningDirectory, "tests\ES")
## add path
sys.path.append(runningDirectory)
sys.path.append(testDirectory)

from main_lr import main_air_lr
    
    
if __name__ == "__main__":
    # simple test of functions : add more tests
    params = dict()
    main_air_lr(params)
    
    