# -*- coding: utf-8 -*-
"""
Created on Sat Mar 17 16:34:28 2018

@author: NILESH
"""
import numpy
def find_max_predicted_score_for_class(X,all_learned_parameters):
    hypothesis_mat = numpy.matmul(X,numpy.transpose(all_learned_parameters))
    probablistic_hypthesis = 1/(1+numpy.exp(-1*hypothesis_mat))
    class_output = numpy.argmax(probablistic_hypthesis,axis=1)+1
    return class_output    
