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
        print(outImage.shape)
        return outImage
    
    except ValueError:
        print('Zero Padded size must be larger than the original image size!')
        

#image reduction by average
def imgDownSize(inImage,newSize=64,method='mean'): 
    outImage=np.zeros((newSize,newSize))
    ratio=int(np.shape(inImage)[0]/newSize)
    if (ratio % 2)==0:
        for i in range(0,newSize):
            for j in range(0,newSize):
                window=np.zeros((ratio,ratio))
                window=inImage[ratio*i:ratio*(i+1),ratio*j:ratio*(j+1)]
                if method=='mean':
                    outImage[i,j]=np.mean(window)
        return outImage
    else:
        print('The new size must be multiple of 2! Empty matrix will be returned!')     
        return []         
    
#Cut image so that it will be multiples of 2
def imgTrim(inImage,newSize=1024):     
    outImage=np.zeros((newSize,newSize))
    offX=int((np.shape(inImage)[0]-np.shape(outImage)[0])/2)
    offY=int((np.shape(inImage)[1]-np.shape(outImage)[1])/2)                
    outImage=inImage[offX:(offX+np.shape(outImage)[0]),offY:(offY+np.shape(outImage)[1])]            
    return outImage
    
            
            