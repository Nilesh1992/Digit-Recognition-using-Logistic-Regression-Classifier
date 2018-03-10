# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 17:11:11 2018

@author: NILESH
"""

import numpy
#Input: The the input matrix of size n*number_of_features  and parameters is of size
# 1 * number_of_features
#output: hypothesis_output is of size n*1
#Sigmod function: 1/1+e^(-(parameters*input_features)) 
def get_sigmoid_hypothesis(parameters,input_matrix):
    try:
        total_input, features = numpy.shape(input_matrix)
        hypothesis_output = numpy.zeros((total_input,1),dtype=numpy.float64)
        for i in range(0,total_input):
            hypothesis_output_intermidiate = numpy.matmul(input_matrix[i],numpy.transpose(parameters))
            hypothesis_output[i] = (1/(1+numpy.exp(-hypothesis_output_intermidiate)))            
        return hypothesis_output    
    except Exception:
        print("Issue in getting hypothesis\n 1)Please check the dimensions ")
            
#out = get_sigmoid_hypothesis([[1,2,3]],[[-2,-3,1],[-2,-3,1]])
#print(out)