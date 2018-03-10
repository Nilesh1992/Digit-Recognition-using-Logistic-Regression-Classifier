# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 21:31:04 2018

@author: NILESH
"""
import gradient_descent
import get_learning_data
import os
import numpy
import matplotlib.pyplot as plt
import predict_class
#This module is to visualize the data in 2-d plot
object_for_data_set = get_learning_data.ReadTextDataSet()
file_path = os.getcwd() + "\ex2data1.txt"
traning_input = object_for_data_set.get_data_in_range_from_the_data_set(file_path,2,0)
traning_output = object_for_data_set.get_data_in_range_from_the_data_set(file_path,2,1)
length = len(traning_input)
new_feature = numpy.ones((length,1),dtype=numpy.float64)
Y = traning_output
X = numpy.append(new_feature,traning_input,axis=1)
number_of_parameters = len(X[0])
theta = numpy.zeros((1,number_of_parameters),dtype=numpy.float64)\
#theta=[[-15.39517866,0.12825989,0.12247929]]
#theta = [[-19.08468795,0.15767916,0.15228738]]
#theta = [[-21.06746245,0.17350979,0.16833432]]
#theta = [[-22.30213181,0.18337323,0.17832949]]
iterations = 1000000
learning_rate = 0.001
#J_history,theta = gradient_descent.gradient_descent_for_function_minimization(iterations,learning_rate,X,theta,Y)
#print(J_history)
#print(theta)
#*********************Theta values after lerning 
theta = [[-24.932759,0.204406,0.199616]]
new_instances = [[1,45,85]]
predicted_class_for_instances = predict_class.get_class_for_input_instances(new_instances,theta)
print(predicted_class_for_instances)