# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 19:54:09 2017

@author: edwin11k
"""

import numpy as np
import os
import glob
from PIL import Image


class ReadPNG(object):
    #this class read PNG file and save for other purpose
    def __init__(self,folderPath=os.getcwd(),type1='imageType'):
        png=[]
        dirPath=folderPath+"/"+type1
        for image_path in glob.glob(dirPath+"/*.png"):
            png.append(Image.open(image_path)) 
        self.img=png
        self.type=type1
        self.template=Image.open(folderPath+"/0.png")
        
    def displayPNGInfo(self):
        print('---------------------------------------------------')
        print('Image category:',self.type)
        print('Number of image:',len(self.img))
        print('Size of image:',np.shape(self.img[0]))
    
    
    
    #zero pad the image set..Input layers should be multiples of 2 s
    # 4 8 16 32 64 128 256 512 1024 2048 4096
    # if it's smaller than original, it reduces into the new size 
    def zeroUnipadTypeSet(self,size=1024):
        try:
            img2=[]
            for png in self.img:
                png2=np.zeros((size,size))
                offX=int((np.shape(png2)[0]-np.shape(png)[0])/2)
                offY=int((np.shape(png2)[1]-np.shape(png)[1])/2)
                png2[offX:(offX+np.shape(png)[0]),offY:(offY+np.shape(png)[1])]=png
                img2.append(png2)
            self.img=img2
        except ValueError:
            print('Warning: Original image will fit into new size & some information will be lost!')
            for png in self.img:
                png2=np.zeros((size,size))
                offX=int((np.shape(png)[0]-np.shape(png2)[0])/2)
                offY=int((np.shape(png)[1]-np.shape(png2)[1])/2)                
                png2=png[offX:(offX+np.shape(png2)[0]),offY:(offY+np.shape(png2)[1])]
                img2.append(png2)
            self.img=img2
            
 
    #need to handle template file separately
    def imgDownsize(self,newSize=64):
        from utils import imgDownSize
        img2=[]
        for png in self.img:
            png2=imgDownSize(png,newSize=newSize,method='mean')
            img2.append(png2)
        self.img=img2
        