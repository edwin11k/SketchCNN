# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 13:01:27 2017

@author: edwin11k
"""
from __future__ import print_function
import numpy as np
from CNNS.sketch.rnsPNG import RnsPNG
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras import backend as K 
import copy
import heapq
from keras.layers.normalization import BatchNormalization
   
def sketchTest(scale=[1],rot=[0]):

    print('Sketch Recognition!')
    epoch=1000
    bEpoch=1000
    imageSize=16
    testPercent1=0.02
    DropOutRate=0.3
    bTrigger=0.05
    imgR,imgC=imageSize,imageSize
    
    
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
    typeZ=['zebra']
    
    type1=typeA+typeB+typeC+typeD+typeE+typeF+typeG+typeH+typeI+typeK+typeL+typeM+typeN+typeO+typeP+typeR+typeS+typeT+typeU+typeV+typeW+typeZ
    #  type1=typeV
    
    #type1=['nose','violin']
    num_class=len(type1);#tType='airplane'
    #print('Validation for',tType)
    fileString='/home/codas/Documents/Edwin/sketchR/result/epoch'+str(epoch)+'bEpoch'+str(bEpoch)+'size'+str(imageSize)+'testPercent'+str(testPercent1)+'Drop'+str(DropOutRate)+'scale'+str(scale)+'rot'+str(rot)+'NumClass'+str(num_class)+'bTrigger'+str(bTrigger)+'.txt'
    print(fileString)
    print('Begin reading files for training Convolution Neural Network!')
    print()
    trainArgs=[];testArgs=[];trainDCY=[];testDCY=[];labelIndex=0
    

        
    for mem in type1:
        D=RnsPNG(dirPath='/home/codas/Documents/Edwin/sketchR/ELCode/CNNS/Data/sketches_png/png',type1=mem,zPad=1112,trimSize=1024,cSize=imgR,angleV=rot,scaleV=scale)
        D.dataStackForCNN(normalize=1,floatation=1,inverse=1)
        D.extractTrainTestfile(testPercent=testPercent1,label=labelIndex)
        trainArgs.append(D.srImgTrainX);testArgs.append(D.srImgTestX);
        trainDCY=trainDCY+D.srImgTrainY;testDCY=testDCY+D.srImgTestY;  
        labelIndex+=1
        print(mem) 
       
    trainDC=np.concatenate(trainArgs,axis=0);trainDC=trainDC.reshape(trainDC.shape[0],imgR,imgC,1)
    testDC=np.concatenate(testArgs,axis=0);testDC=testDC.reshape(testDC.shape[0],imgR,imgC,1)
    trainArgs=[];testArgs=[]
    trainDCY = keras.utils.to_categorical(trainDCY, num_class)
    testDCY2 = keras.utils.to_categorical(testDCY, num_class)
    
    
    print('About to go to the CNN!')
    model=Sequential()
    model.add(Conv2D(10,kernel_size=(3,3),activation='relu',input_shape=(imgR,imgC,1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(10,(3,3),activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(DropOutRate))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(DropOutRate))
    model.add(Dense(128,activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(DropOutRate))   
    model.add(Dense(num_class,activation='softmax'))
    
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
    print('Modeling fitting initiate!')
    model.fit(trainDC,trainDCY,batch_size=128,epochs=epoch,verbose=1,validation_data=(testDC,testDCY2))

    # test score from original CNN
    score=model.evaluate(testDC,testDCY2,verbose=1)
    print('Test loss:',score[0])
    print('Test accuracy:',score[1])    
    
    #fileString='C:/Users/edwin/Documents/Sketch/results/epoch'+str(epoch)+'size'+str(imageSize)+'testPercent'+str(testPercent1)+'scale'+str(scale)+'rot'+str(rot)+'NumClass'+str(num_class)+'.txt'
    
    f=open(fileString,'w')
    f.write('------------- CNN original result ---------------\n')
    f.write('Test loss:'+str(score[0])+'\n')
    f.write('Test accuracy:'+str(score[1])+'\n')
    f.write('-------------------------------------------------\n')    
    
    correct=0;total=0;pos=0;neg=0;
    for testIndex in range(testDC.shape[0]):
        typeI=[];runOffFlag=False
        testSlice=np.zeros((1,imgR,imgC,1));testSlice[0,:,:,0]=testDC[testIndex,:,:,0]
        fr=model.predict_proba(testSlice)[0]
        max2=heapq.nlargest(2, range(len(fr)), key=fr.__getitem__)
        fValue=fr[max2[0]];sValue=fr[max2[1]]
        typeI=[type1[i] for i in max2]
        print(typeI)
        print(fr[max2])
        if (fValue-sValue<bTrigger):
            print('Try Runoff!')
            runOffFlag=True
            typeF=binaryCNN(testSlice,typeI,testPercent1,imgR,imgC,bEpoch,rot,scale,DropOutRate)
        else:
            typeF=typeI[0]
        print('Trained Solution:'+typeF);
        print('Correct Solution:'+type1[testDCY[testIndex]]);
        if type1.index(typeF)==testDCY[testIndex]:
            correct+=1
        total+=1
        
        if type1.index(typeF)==testDCY[testIndex] and typeF==typeI[1] and runOffFlag:
            pos+=1
        if type1.index(typeF)!=testDCY[testIndex] and typeF==typeI[0] and runOffFlag:
            neg+=1
        
        
        
        print ('positive: {}, negative:{} '.format(str(pos),str(neg)))
        print ('correct: {}, total:{} , validation:{}'.format(correct,total,correct/total))
    
    f.write('------------- CNN Improved ---------------\n')
    f.write('Test accuracy:'+str(correct/total)+'\n')
    f.write('------------------------------------------\n')        
    f.close()


def binaryCNN(testSample,types,testPercent1,imgR,imgC,epoch,rot,scale,DropOutRate):
    trainArgs1=[];trainDCY1=[];labelIndex1=0
    for mem in types:
        D=RnsPNG(dirPath='/home/codas/Documents/Edwin/sketchR/ELCode/CNNS/Data/sketches_png/png',type1=mem,zPad=1112,trimSize=1024,cSize=imgR,angleV=rot,scaleV=scale)
        D.dataStackForCNN(normalize=1,floatation=1,inverse=1)
        D.extractTrainTestfile(testPercent=testPercent1,label=labelIndex1)
        trainArgs1.append(D.srImgTrainX);trainDCY1=trainDCY1+D.srImgTrainY 
        labelIndex1+=1
          
    trainDC=np.concatenate(trainArgs1,axis=0);trainDC=trainDC.reshape(trainDC.shape[0],imgR,imgC,1)
    trainDCY1 = keras.utils.to_categorical(trainDCY1,2)
        
    model=Sequential()
    model.add(Conv2D(10,kernel_size=(3,3),activation='relu',input_shape=(imgR,imgC,1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Conv2D(10,(3,3),activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(DropOutRate))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(DropOutRate))
    model.add(Dense(128,activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(DropOutRate))   
    model.add(Dense(2,activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
    print('Binary Runoff Decision!!')
    model.fit(trainDC,trainDCY1,batch_size=128,epochs=epoch,verbose=0)
    
    if model.predict_classes(testSample)==0:
        return types[0]
    else:
        return types[1]
     

    
sketchTest()


