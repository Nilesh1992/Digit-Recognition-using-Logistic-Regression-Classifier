# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 23:18:03 2018

@author: NILESH
"""
import numpy
import sigmoid
def get_class_for_input_instances(input_matrix,learned_parameters):
    try:
        hypothesis = sigmoid.get_sigmoid_hypothesis(input_matrix,learned_parameters)
        no_of_instances = len(hypothesis)
        class_for_instance = numpy.zeros((no_of_instances,1),dtype=numpy.float64)
        threshold = 0.5 #This the mininum prbablity one should satisfy to classify as positive class. This threshold depends on problem in hand
        for i in range(0,no_of_instances):
            if(hypothesis[i][0]>threshold):
                class_for_instance[i][0] = 1
            else:
                class_for_instance[i][0] = 0   
        return class_for_instance    
    except Exception:
        print("Some issue while predecting class for the input instances")
            