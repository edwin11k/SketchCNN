# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 13:01:27 2017

@author: edwin11k
"""
from __future__ import print_function
import numpy as np
from CNNS.Data.rnsPNG import RnsPNG
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras import backend as K 

   
def sketchTest(scale=[1],rot=[0]):

    print('Sketch Recognition!')
    epoch=100
    imageSize=64
    imgR,imgC=imageSize,imageSize
    print('hello!')
    #D=RnsPNG(dirPath='/home/codas/Documents/Edwin/sketchR/ELCode/CNNS/model/dogNcat',type1='dog',zPad=1112,angleV=rot,scaleV=scale)
    #C=RnsPNG(dirPath='/home/codas/Documents/Edwin/sketchR/ELCode/CNNS/model/dogNcat',type1='cat',zPad=1112,angleV=rot,scaleV=scale)
    typeA=['airplane','alarm clock','angel','ant','apple','arm','armchair','ashtray','axe']
    typeB=['backpack','banana','barn','baseball bat','basket','bathtub','bear','bed','bee','beer-mug','bench','bicycle','binoculars','blimp','book','bookshelf','boomerang','bottle opener','bowl','brain','bread','bridge','bulldozer','bus','bush','butterfly']
    typeC=['cabinet','cactus','cake','calculator','camel','candle','cannon','canoe','car','carrot','castle','cat','cell phone','chair','chandelier','church','cigarette','cloud','comb','computer monitor','computer-mouse','couch','cow','crab','crane','crocodile','crown','cup']
    typeD=['diamond','dog','dolphin','donut','door','door handle','dragon','duck']
    typeE=['ear','elephant','envelope','eye','eyeglasses']
    typeF=['face','fan','feather','fire hydrant','fish','flashlight','floor lamp','flower with stem','flying bird','flying saucer','foot','fork','frying-pan']
    typeG=['giraffe','grapes','grenade','guitar']
    typeH=['hamburger','hammer','hand','harp','hat','head','head-phones','hedgehog','helicopter','helmet','horse','hot air balloon','hot-dog','hourglass','house','human-skeleton','ice-cream-cone']
    typeI=['ipod']
    typeK=['kangaroo','key','keyboard','knife']
    typeL=['ladder','laptop','leaf','lightbulb','lighter','lion','lobster','loudspeaker']
    typeM=['mailbox','megaphone','mermaid','microphone','microscope','monkey','moon','mosquito','motorbike','mouse','mouth','mug','mushroom']
    typeN=['nose']
    typeO=['octopus','owl']
    typeP=['palm tree','panda','paper clip','parachute','parking meter','parrot','pear','pen','penguin','person sitting','piano','pickup truck','pig','pigeon','pineapple','pipe','pizza','potted plant','power outlet','present','pretzel','pumpkin','purse']
    typeR=['race car','radio','rainbow','revolver','rifle','rollerblades','rooster']
    typeS=['sailboat','santa claus','satellite','satellite dish','saxophone','scissors','scorpion','screwdriver','sea turtle','seagull','shark','sheep','ship','shoe','shovel','skateboard','skull','skyscraper','snail','snake','snowboard','snowman','socks','space shuttle','speed-boat','spider','sponge bob','spoon','squirrel','standing bird','stapler','strawberry','streetlight','submarine','suitcase','sun','suv','swan','sword','syringe']
    typeT=['table','tablelamp','teacup','teapot','teddy-bear','telephone','tennis-racket','tent','tiger','tire','toilet','tomato','tooth','toothbrush','tractor','traffic light','train','tree','trombone','trousers','truck','trumpet','t-shirt','tv']
    typeU=['umbrella']
    typeV=['van','vase','violin']
    typeW=['walkie talkie','wheel','wheelbarrow','windmill','wine-bottle','wineglass']
    typeZ=['zebra','zmp']
    
    type1=typeA+typeB+typeC+typeD+typeE+typeF+typeG+typeH+typeI+typeK+typeL+typeM+typeN+typeO+typeP+typeR+typeS+typeT+typeU+typeV+typeW+typeZ
    num_class=2;tType='airplane'
    print('Validation for ',tType)
    print('Begin reading files for training Convolution Neural Network!')
    trainArgs=[];testArgs=[];trainDCY=[];testDCY=[];labelIndex=0
    
    for mem in type1:
        D=RnsPNG(dirPath='C:/Users/edwin11k/Documents/MQF/SketchR/Download/sketches_png/png',type1=mem,zPad=1112,trimSize=1024,cSize=imgR,angleV=rot,scaleV=scale)
        D.dataStackForCNN(normalize=1,floatation=1,inverse=1)
        if mem==tType:
            labelIndex=1
            D.divideTrainTest(testPercent=0.2,label=labelIndex)
        else:
            labelIndex=0
            D.divideTrainTest(testPercent=0,label=labelIndex)
        
        trainArgs.append(D.srImgTrainX);testArgs.append(D.srImgTestX);
        trainDCY=trainDCY+D.srImgTrainY;testDCY=testDCY+D.srImgTestY;       
        print(mem)
    trainDC=np.concatenate(trainArgs,axis=0);trainDC=trainDC.reshape(trainDC.shape[0],imgR,imgC,1)
    testDC=np.concatenate(testArgs,axis=0);testDC=testDC.reshape(testDC.shape[0],imgR,imgC,1)
    trainArgs=[];testArgs=[]
    trainDCY = keras.utils.to_categorical(trainDCY, num_class)
    testDCY = keras.utils.to_categorical(testDCY, num_class)
    
    
    print('About to go to the CNN!')
    model=Sequential()
    
    model.add(Conv2D(4,kernel_size=(3,3),activation='relu',input_shape=(imgR,imgC,1)))
    print('First convolution layer')
    model.add(Conv2D(4,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(num_class,activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
    print('Modeling fitting initiate!')
    model.fit(trainDC,trainDCY,batch_size=128,epochs=epoch,verbose=1,validation_data=(testDC,testDCY))
    score=model.evaluate(testDC,testDCY,verbose=1)
    print('Test loss:',score[0])
    print('Test accuracy:',score[1])
    
    
sketchTest()