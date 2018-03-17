# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 22:50:48 2018

@author: NILESH
"""


import get_learning_data
import os
import numpy
import matplotlib.pyplot as plt
import sigmoid
import feature_map
import predict_class
#This module is to visualize the data in 2-d plot
def find_hypothesis(x,y,theta):
    mapped = feature_map.create_new_features_with_any_power([[x,y]],6)[1][:]
    feature = numpy.append([[1]],[mapped],axis=1)
    value = predict_class.get_class_for_input_instances(feature,theta)
    return value
object_for_data_set = get_learning_data.ReadTextDataSet()
file_path = os.getcwd() + "\ex2data2.txt"
input_data = object_for_data_set.get_data_in_range_from_the_data_set(file_path,2,0)
output_data = object_for_data_set.get_data_in_range_from_the_data_set(file_path,2,1)
positive_class = numpy.array([[0,0]],dtype=numpy.float64)
negative_class = numpy.array([[0,0]],dtype=numpy.float64)
number_of_traning_data = len(output_data)
for i in range(0,number_of_traning_data-1):
    if(output_data[i][0]==1.0):
        positive_class = numpy.append(positive_class,[input_data[i]],axis=0)
    else:
        negative_class = numpy.append(negative_class,[input_data[i]],axis=0)
data_x_axis = negative_class.T[0][1::] #negative_class.T[0][1::] this give use the data in range 1 to m
data_y_axis = negative_class.T[1][1::]
plt.plot(data_x_axis,data_y_axis,'yo')
data_x_axis = positive_class.T[0][1::] #positive_class.T[0][1::] this give use the data in range 1 to m
data_y_axis = positive_class.T[1][1::]
plt.plot(data_x_axis,data_y_axis,'r+')
#*******************Regularized_parameter lamda =1 **********************
#Just right decsion boundry
theta = [[ 1.27205051,0.62495932,1.1807451,-2.01880313,-0.91623058,-1.4293666
   ,0.12415851, -0.36569678, -0.35783018 ,-0.17477476, -1.45825434, -0.05183912,
  -0.61565861, -0.27441207, -1.19279579, -0.24195793, -0.20633987, -0.04550701,
  -0.27761751, -0.2953999,  -0.45718705, -1.0434562 ,  0.02712759, -0.2926233,
   0.01507605, -0.32739184, -0.14370986, -0.92587262]]
x_range = numpy.arange(numpy.floor(numpy.min(input_data)),numpy.ceil(numpy.max(input_data)),0.01)
y_grid,x_grid = numpy.meshgrid(x_range,x_range)
row,col = numpy.shape(y_grid)
z = numpy.zeros((row,col),dtype=numpy.float64)
for i in range(0,row):
    for j in range(0,col):
        z[i][j] = find_hypothesis(x_grid[i][j],y_grid[i][j],theta)
plt.contour(x_grid, y_grid, z,colors='blue')
#*******************Regularized_parameter lamda =1000 **********************
#underfitting of decsion boundry
theta = [[ 0.02187771,-0.01748172 , 0.00571079, -0.05516895, -0.01314877, -0.03859858
 , -0.01846356, -0.00773219 ,-0.00892429, -0.02280452, -0.04343846 ,-0.00235623,
  -0.01415612, -0.00349508, -0.04143588, -0.02100593 ,-0.00471917, -0.00359131,
  -0.00632226, -0.00502441 ,-0.03197676, -0.03416335, -0.00107629, -0.00702615
  ,-0.00038506, -0.0079823,  -0.00154779 ,-0.04108677]]
row,col = numpy.shape(y_grid)
z_1 = numpy.zeros((row,col),dtype=numpy.float64)
for i in range(0,row):
    for j in range(0,col):
        z_1[i][j] = find_hypothesis(x_grid[i][j],y_grid[i][j],theta)
plt.contour(x_grid, y_grid, z_1,colors='red')
#****************Regularized_parameter lamda =0 ****************************
#over fitting of decsion boundry
theta = [[26.4323705539509,37.9557942526205,57.7845542472037,-256.851774985232,-125.719907230075
,-132.836698904681,-254.895297761343,-511.227109139429,-425.616097747407,-287.478278586376
,835.534508527139,875.222586281607,1286.08444259639,608.624864363425,328.032068324908,407.075683660401
,1031.65338149182,1562.67426569084,1689.17553657007,1006.38734047474,434.387923247629,-904.087844767451
,-1527.75987968412,-2702.94644154411,-2679.00072843319,-2628.70593619858,-1246.87311324025
,-441.207454035653]]
row,col = numpy.shape(y_grid)
z_1 = numpy.zeros((row,col),dtype=numpy.float64)
for i in range(0,row):
    for j in range(0,col):
        z_1[i][j] = find_hypothesis(x_grid[i][j],y_grid[i][j],theta)
plt.contour(x_grid, y_grid, z_1,colors='green')