# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 15:53:44 2018

@author: NILESH
"""

import numpy as nu
import cost_function_for_logistic_regression_with_regularization
import sigmoid
def gradient_descent_for_function_minimization_and_remove_overfitting(iteration,learning_rate,X,theta,Y,lamda):
    J_history = nu.zeros((iteration,1),dtype=nu.float64)
    m = len(X)
    for i in range(0,iteration):
        regularized_theta = (lamda/m)*theta
        regularized_theta[0][0] = 0 #We are not regulatizing the theta0
        J_history[i,0] = cost_function_for_logistic_regression_with_regularization.cost_function_for_data_regularized(X,Y,theta,lamda)
        hypothesis_output = sigmoid.get_sigmoid_hypothesis(theta,X)
        output_vertor = (hypothesis_output - Y)
        derivative_with_respect_theta = (1/m) * nu.matmul(nu.transpose(X),output_vertor)
        theta = theta - (learning_rate*(nu.transpose(derivative_with_respect_theta) + regularized_theta))
    return J_history,theta    
  