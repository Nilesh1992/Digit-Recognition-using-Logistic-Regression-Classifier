# -*- coding: utf-8 -*-
import numpy

def get_accuracy_for_logistic_regression(Y,predicted_class):
    Y = Y.astype(numpy.uint8)
    predicted_class = predicted_class.astype(numpy.uint8)
    length = len(Y)
    counter = 0
    for i in range(0,length):
        if(Y[:][i]==predicted_class[:][i]):
            counter = counter + 1
    return ((counter/length)*100)                

