# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 23:09:07 2018

@author: NILESH
"""

import scipy.io as spio
import os
import numpy
import matplotlib.pyplot as plt
import random
file_path = os.getcwd() + "\ex2data3.mat"
mat = spio.loadmat(file_path, squeeze_me=True) #This is to load the matrix
X = mat['X']
#This gives the better visualization of numbers 100 randamly selected numbers
image_width = 20
image_height = 20
row = 10
col = 10
total = (row*col + 1)
length = len(X)
fig= plt.figure(figsize=(8,8))
for i in range(1,total):
        random_integer = random.randint(0,length) 
        img = numpy.reshape((X[:][random_integer:random_integer+1]),(20,20),'F')     
        fig.add_subplot(row, col,i)
        plt.imshow(img)