# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 11:08:54 2017

@author: edwin11k
"""

import numpy as np
from .Data.rnsPNG import RnsPNG

def dataLoadTest(scale=[0.2,0.5,0.7],rot=[90,180,270]):

    print('data loading test')
    A=RnsPNG(dirPath='C:/Users/edwin11k/Documents/MQF/SketchR/Download/sketches_png/png',type1='airplane')
    A.displayPNGInfo()