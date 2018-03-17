# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 15:10:30 2018

@author: NILESH
"""

import gradient_descent_regularized_version
import get_learning_data
import os
import numpy
import matplotlib.pyplot as plt
import predict_class
import feature_map
object_for_data_set = get_learning_data.ReadTextDataSet()
file_path = os.getcwd() + "\ex2data2.txt"
traning_input = object_for_data_set.get_data_in_range_from_the_data_set(file_path,2,0)
traning_output = object_for_data_set.get_data_in_range_from_the_data_set(file_path,2,1)
length = len(traning_input)
new_feature = numpy.ones((length,1),dtype=numpy.float64)
Y = traning_output
X = feature_map.create_new_features_with_any_power(traning_input,6)[1::]
X = numpy.append(new_feature,X,axis=1)
number_of_parameters = len(X[0])
theta = numpy.zeros((1,number_of_parameters),dtype=numpy.float64)
#For this specific example no need to feature scaling as we don't have much diff in values in variabels
iterations = 1000000
learning_rate = 15
lamda = 0
theta = numpy.array([[   6.73339769 ,   4.52042597 ,   5.22466778 , -60.2122472 ,  -22.20437028,
   -16.23856421 , -38.59477778 , -10.29347681 ,  11.37803328 ,  -8.1172086,
   210.71771105 , 111.50162272 , 170.24273199  , 44.82087843  , 10.27826732,
    78.96467777,   49.78589153 ,  38.18032645 ,  18.99716614 ,  13.74916887,
    33.14814019, -252.08947698, -199.52389155, -256.84424393 , -83.57625874,
  -244.39263324, -116.659401 ,   -38.49755678]],dtype=numpy.float64)
#learned_parameters_with_lamda_1 = [[ 1.27205051  0.62495932  1.1807451  -2.01880313 -0.91623058 -1.4293666
#   0.12415851 -0.36569678 -0.35783018 -0.17477476 -1.45825434 -0.05183912
#  -0.61565861 -0.27441207 -1.19279579 -0.24195793 -0.20633987 -0.04550701
#  -0.27761751 -0.2953999  -0.45718705 -1.0434562   0.02712759 -0.2926233
#   0.01507605 -0.32739184 -0.14370986 -0.92587262]]
#learned_parameters_with_lamda_100 = [[ -2.76990259e-02  -2.11716688e-03   2.19216806e-04  -5.67619963e-03
  # -1.37472770e-03  -4.11277687e-03  -2.07831422e-03  -8.26757969e-04
  # -9.44899445e-04  -2.58908937e-03  -4.50270751e-03  -2.65597976e-04
  # -1.46171855e-03  -3.73222417e-04  -4.43132777e-03  -2.28289540e-03
  # -4.96098428e-04  -3.86839267e-04  -6.66372893e-04  -5.21777233e-04
  # -3.50495707e-03  -3.56609728e-03  -1.27654450e-04  -7.23875275e-04
  # -4.85933015e-05  -8.30775746e-04  -1.67100388e-04  -4.41422935e-03]]
J_history,theta = gradient_descent_regularized_version.gradient_descent_for_function_minimization_and_remove_overfitting(iterations,learning_rate,X,theta,Y,lamda)
print(J_history)
print(theta)