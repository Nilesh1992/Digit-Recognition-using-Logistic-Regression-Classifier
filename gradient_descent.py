# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 19:44:50 2018

@author: NILESH
"""
import numpy as nu
import cost_function_for_logistic_regression_without_regularization
import sigmoid
def gradient_descent_for_function_minimization(iteration,learning_rate,X,theta,Y):
    J_history = nu.zeros((iteration,1),dtype=nu.float64)
    m = len(X)
    for i in range(0,iteration):
        J_history[i,0] = cost_function_for_logistic_regression_without_regularization.cost_function_for_data_unregularized(X,Y,theta)
        hypothesis_output = sigmoid.get_sigmoid_hypothesis(theta,X)
        output_vertor = (hypothesis_output - Y)
        derivative_with_respect_theta = (1/m) * nu.matmul(nu.transpose(X),output_vertor)
        theta = theta - (learning_rate*(nu.transpose(derivative_with_respect_theta)))
    return J_history,theta    
        
        
    