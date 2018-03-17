# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 14:14:09 2018

@author: NILESH
"""

import numpy
#Note it is specific to 2-variable feature
def create_new_features_with_any_power(traning_set,power):
    try:
        number_of_new_features = int(((power + 1)*(power+2)/2) - 1)  #This just n(n+1)/2 variation
        number_traning_examples = len(traning_set)
        new_feature_array = numpy.zeros((1,number_of_new_features),dtype=numpy.float64)
        temp = numpy.zeros((1,number_of_new_features),dtype=numpy.float64)
        for k in range(0,number_traning_examples):
            counter = 0
            x1 = traning_set[k][0]
            x2 = traning_set[k][1]
            for i in range(1,power+1):    
                for j in range(0,i+1):
                    temp[0][counter] = numpy.power(x1,(i-j))*numpy.power(x2,j)
                    counter = counter + 1
            new_feature_array = numpy.append(new_feature_array,temp,axis = 0)        
        return new_feature_array    
    except Exception:
        print("Some issue while creating new features form the input traning set")     