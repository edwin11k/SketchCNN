# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 19:07:35 2017

@author: edwin11k
"""

import matplotlib.pyplot as plt
from scipy import misc
import numpy as np
from PIL import Image

A=Image.open('1.png')
plt.imshow(np.asarray(A))
#make it small
B = A.resize((int(A.size[0]/2), int(A.size[1]/2)), Image.NEAREST)
B2 = B.rotate(45)

#making mask! '0.png'
M=Image.open('0.png')
MS= M.resize((int(M.size[0]/2), int(M.size[1]/2)), Image.NEAREST)
MSR=MS.rotate(45)
dM=np.asarray(MS)-np.asarray(MSR)



#zero pad the image
B2=np.asarray(B2)+dM
B3=np.pad(B2,((278,278),(278,278)),mode='constant',constant_values=255)
##resize test
plt.imshow(np.asarray(B2))
