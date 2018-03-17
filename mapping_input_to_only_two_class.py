# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 23:55:15 2018

@author: NILESH
"""
#This give us a binary map the dataset as just in positive class 
#negative class
def binary_class_mapping(Y,positive_class):
    return ((Y==positive_class)*1).astype(float)
    
    