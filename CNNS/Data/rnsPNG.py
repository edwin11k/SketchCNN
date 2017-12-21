# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 21:38:50 2017

@author: edwin11k
"""

import numpy as np
from .readPNG import ReadPNG
import os
from PIL import Image
#import matplotlib.pyplot as plt
from .utils import imgDownSize,imgTrim

class RnsPNG(ReadPNG):
    # path should be the path of the data set folder, filepath=path/type
    def __init__(self,dirPath=os.getcwd(),type1='imageType',zPad=1024,trimSize=1024,cSize=64,angleV=[0,90,180,270],scaleV=[1,0.75,0.5,0.25]):
        super(RnsPNG,self).__init__(dirPath,type1)
        self.srImg=[];self.scale=scaleV;self.angle=angleV
        for img in self.img:
            sImg=[]
            for scale in scaleV:
                rImg=[]
                for angle in angleV:
                    #scale the image by scale factor
                    imgS=img.resize((int(np.shape(img)[0]*scale),int(np.shape(img)[1]*scale)),Image.NEAREST)
                    #rotate the image by angle 
                    imgSR=imgS.rotate(angle)
                    
                    #####################################################################################################
                    #template adjustment
                    MS=self.template.resize((int(np.shape(self.template)[0]*scale),int(np.shape(self.template)[1]*scale)),Image.NEAREST)
                    MSR=MS.rotate(angle)
                    dM=np.asarray(MS)-np.asarray(MSR)
                    imgSR=np.asarray(imgSR)+dM
                    ######################################################################################################
                    #zero pad so that it will be like an orignal size
                    offX=np.shape(img)[0]-np.shape(imgSR)[0]+(zPad-np.shape(img)[0])
                    offY=np.shape(img)[1]-np.shape(imgSR)[1]+(zPad-np.shape(img)[1])
                    imgSRP=np.pad(imgSR,((int(offX/2),offX-int(offX/2)),(int(offY/2),offY-int(offY/2))),mode='constant',constant_values=255)
                    #trim image so that it will be multiple of 2
                    imgSRP=imgTrim(imgSRP,newSize=trimSize)
                    #contract to smaller size
                    imgSRP=imgDownSize(imgSRP,newSize=cSize,method='mean')               
                    rImg.append(imgSRP)
                sImg.append(rImg)
            self.srImg.append(sImg)



        
    #restructure data structure for CNN,, may not need it future
    def dataStackForCNN(self,floatation=1,normalize=1,inverse=0):
        #image needs to be input as 3d numpy stack      
        srImgStack=np.zeros((len(self.srImg)*len(self.scale)*len(self.angle),(self.srImg[0][0][0]).shape[0],(self.srImg[0][0][0]).shape[1]))
        index=0;
        for imgTrain in self.srImg:
            for scaleMem in imgTrain:
                for rotMem in scaleMem:
                    if floatation==True:
                        rotMem=rotMem.astype('float32')
                    if normalize==True:
                        rotMem/=255
                    if inverse==True:
                        rotMem=abs(rotMem-rotMem.max())
                    srImgStack[index,:,:]=rotMem
                    index+=1
        self.srImg=srImgStack
        
        
    def divideTrainTest(self,testPercent=0.1,remove_original=True,label=0):
        permIndex=np.random.permutation(len(self.srImg))
        trainIndex=permIndex[0:int(len(self.srImg)*(1-testPercent))]
        trainThick=len(trainIndex)
        testThick=len(self.srImg)-trainThick
      #  testIndex=permIndex[int(len(self.img)*(1-testPercent)):-1]
        self.srImgTrainX=np.zeros((trainThick,(self.srImg[0]).shape[0],(self.srImg[0]).shape[1]))
        self.srImgTestX=np.zeros((testThick,(self.srImg[0]).shape[0],(self.srImg[0]).shape[1]))
        self.srImgTrainY=[]
        self.srImgTestY=[]
        
        for index in permIndex:
            if index in trainIndex:
                self.srImgTrainX=self.srImg[0:trainThick,:,:]
                self.srImgTrainY.append(label)
            else:
                self.srImgTestX=self.srImg[trainThick:,:,:]
                self.srImgTestY.append(label)
        if remove_original==True:
            self.srImg=[]
   
 

         #may be able to view the image, but commented since matplotlib is not installed    
#    def viewImg(self,fileNum=0,scaleNum=0,angleNum=0):
#        plt.imshow(((self.srImg[fileNum])[scaleNum])[angleNum])
        
        
   