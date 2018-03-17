
import get_learning_data
import os
import numpy
import matplotlib.pyplot as plt
import sigmoid
#This module is to visualize the data in 2-d plot
def find_line(x,y):
    value = (-24.932759 + (0.204406*x) + (0.199616*y))
    return value
object_for_data_set = get_learning_data.ReadTextDataSet()
file_path = os.getcwd() + "\ex2data1.txt"
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
theta = [[-24.932759,0.204406,0.199616]] #Linear predection
x_range = numpy.arange(numpy.floor(numpy.min(input_data)),numpy.ceil(numpy.max(input_data)),0.01)
y_grid,x_grid = numpy.meshgrid(x_range,x_range)
plt.plot([92.62,30.05], [30.05,94.14])
#How to calculate this endpoints mathmatically
#After optimizing the parameters with the gradient descent we got 
#-24.932759 + (0.204406*x) + (0.199616*y)
#the hypothesis will be positive class only when
#-24.932759 + (0.204406*x) + (0.199616*y)>=0
#x = 30.05 which is the min in the data set
#we can calculate y >= ((0.204406*x) + 24.932759)/0.199616
#i.e all the values greater than the computed contant will be classified as + class
#Smilarly we can calculate for y=30.05 as well 