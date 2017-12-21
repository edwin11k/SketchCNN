# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:13:40 2017

@author: edwin11k
"""


from __future__ import print_function
import keras
#from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
#from keras import backend as K
import csv
import numpy as np
#import matplotlib.pyplot as plt

with open('C:/Users/edwin11k/Documents/MQF/SketchR/ELCode/CNNS/dataZip/fashionmnist/fashion-mnist_test.csv','r') as testFile:
    testData = csv.reader(testFile, delimiter=',')
    testList=list(testData)
    
fImage=[]  # 28x28 size image
headFlag=True

for imageList in testList:
    if headFlag==True:
        headList=imageList
        headFlag=False
    else:
        newImage=[]
        for mem in imageList:
            newImage.append(int(mem))
        newImage=np.asarray(newImage)
        fImage.append(newImage)

# reshape 28x28
testLabel=[]
imageTest=np.zeros((len(fImage),28,28))
index=0
for image1 in fImage:
    testLabel.append(image1[0])
    temp=np.asarray(image1[1:len(image1)])
    temp=temp.reshape((28,28))
    imageTest[index,:,:]=temp
    index+=1
    
#clear the memory
fImage=[]

with open('C:/Users/edwin11k/Documents/MQF/SketchR/ELCode/CNNS/dataZip/fashionmnist/fashion-mnist_train.csv','r') as trainFile:
    trainData = csv.reader(trainFile, delimiter=',')
    trainList=list(trainData)
    
fImage=[]  # 28x28 size image
headFlag=True

for imageList in trainList:
    if headFlag==True:
        headList=imageList
        headFlag=False
    else:
        newImage=[]
        for mem in imageList:
            newImage.append(int(mem))
        newImage=np.asarray(newImage)
        fImage.append(newImage)

# reshape 28x28
trainLabel=[]
imageTrain=np.zeros((len(fImage),28,28))
index=0
for image1 in fImage:
    trainLabel.append(image1[0])
    temp=np.asarray(image1[1:len(image1)])
    temp=temp.reshape((28,28))
    imageTrain[index,:,:]=temp
    index+=1
        
           
fImage=[] 


batch_size=128
num_classes=10
epochs=12

img_rows,img_cols=28,28

imageTrain=imageTrain.reshape(imageTrain.shape[0],img_rows,img_cols,1)
imageTest=imageTest.reshape(imageTest.shape[0],img_rows,img_cols,1)
input_shape=(img_rows,img_cols,1)   
    
imageTrain=imageTrain.astype('float32')
imageTest=imageTest.astype('float32')
imageTrain/=255
imageTest/=255

y_train = keras.utils.to_categorical(trainLabel, 10)
y_test = keras.utils.to_categorical(testLabel, 10)


model=Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
model.fit(imageTrain,y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(imageTest,y_test))
score=model.evaluate(imageTest,y_test,verbose=1)
print('Test loss:',score[0])
print('Test accuracy:',score[1])