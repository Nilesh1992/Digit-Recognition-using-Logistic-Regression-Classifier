# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 15:56:01 2018

@author: NILESH
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 20:18:05 2018

@author: NILESH
"""
import numpy
import sigmoid
def cost_function_for_data_regularized(input_matrix,output_matrix,theta_parameters,lambda_parameter):
   try:
       total_number_of_traning_example = len(output_matrix)
       current_cost = 0
       regularized_term = 0
       regularized_term = (lambda_parameter/(2*total_number_of_traning_example))*(numpy.sum(numpy.power(theta_parameters[0][1::],2)))
       calculated_hypothesis_cost = sigmoid.get_sigmoid_hypothesis(theta_parameters,input_matrix)    
       for i in range(0,total_number_of_traning_example):
           if(calculated_hypothesis_cost[i] == 1.0):
               calculated_hypothesis_cost[i] = 0.99999999999 #This is trick to keep the value for log fuction non-zero
           if(calculated_hypothesis_cost[i] == 0.0):
               calculated_hypothesis_cost[i] = 0.000000000001
           output_for_current_instance = output_matrix[i]*numpy.log(calculated_hypothesis_cost[i]) + (1.0-output_matrix[i])*numpy.log(1.0-calculated_hypothesis_cost[i])              
           #Cost for a specific traning input 
           current_cost = current_cost + output_for_current_instance
       current_cost = (-(1/total_number_of_traning_example)*current_cost) + regularized_term
       print(current_cost)
       return current_cost
   except Exception:
       print("Isssue in getting cost function otput\n 1) Please check the dimension ")
            