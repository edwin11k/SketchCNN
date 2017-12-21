# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 13:10:50 2017

@author: edwin11k
"""
import numpy as np

def npZeroPad(inImage,padSize=1024):
    try:
        outImage=np.zeros((padSize,padSize))
        offX=int((np.shape(outImage)[0]-np.shape(inImage)[0])/2)
        offY=int((np.shape(outImage)[1]-np.shape(inImage)[1])/2)
        outImage[offX:(offX+np.shape(inImage)[0]),offY:(offY+np.shape(inImage)[1])]=inImage
        return outImage
    except ValueError:
        print('Zero Padded size must be larger than the original image size!')
        

            