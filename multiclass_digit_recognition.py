import scipy.io as spio
import os
import numpy
import gradient_descent_regularized_version
import mapping_input_to_only_two_class
import traning_data_prediction_accuracy
file_path = os.getcwd() + "\ex2data3.mat"
mat = spio.loadmat(file_path, squeeze_me=True) #This is to load the matrix
X = mat['X']
Y = mat['y']
length = len(X)
new_feature = numpy.ones((length,1),dtype=numpy.float64)
#axis = 0 means add in as row
#asis = 1 means add in as column
X = numpy.append(new_feature,X,axis = 1)
no_of_classes = len(set(Y)) #This gives number of classes
number_of_parameters = len(X[0])
iterations = 10000
learning_rate = 0.03
lamda = 1
all_J_cost = numpy.zeros((iterations,1),dtype=numpy.float64)
all_learned_parameters = numpy.zeros((1,number_of_parameters),dtype=numpy.float64)
for i in range(1,no_of_classes+1):
    print(i)
    theta = numpy.zeros((1,number_of_parameters),dtype=numpy.float64)
    output = numpy.reshape(mapping_input_to_only_two_class.binary_class_mapping(Y,i),(length,1))
    J_history,theta = gradient_descent_regularized_version.gradient_descent_for_function_minimization_and_remove_overfitting(iterations,learning_rate,X,theta,output,lamda)
    all_learned_parameters = numpy.append(all_learned_parameters,theta,axis=0)
    all_J_cost = numpy.append(all_J_cost,J_history,axis=1)    
class_hypthesis  = traning_data_prediction_accuracy.find_max_predicted_score_for_class(X,all_learned_parameters[:][1::])    
