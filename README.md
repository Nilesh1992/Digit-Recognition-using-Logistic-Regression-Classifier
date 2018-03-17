# Logistic-regression-classifier for digit recogniton
This contains 3 logistic regression models which are as follows
1) linear logistic regression model for binary classification, where we predict whether a student will get admited to a premier college based on the marks obtained in tow subjects.
2) Non-linear logistic regression model for binary classification. In this we predect a manufactured chip is defective on not defective.
3) A multiclass logistic regression classifier for digit recognition.

==> The data of model 1 can be found in ex2data1.txt, where we can visualize the data on the 2-D plot using the python script Plot_data.py. The data seems to be linearly seprable, So we fit a linear hypothesis function to the data. You can try this yourself that, we can fit a non-linear hypothesis to this as well but keep in mind that it should not overfit the data. So, for this overfitting issue or underfitting issue can be solved using regularizaion.


==> The data of model 2 can be found in ex2data2.txt, where this data can be visulalized using the 2-D plot using plot_data_regularized. The learned decision boundary is plotted using contour plots which are just the 2-D visulization of a 3-D function at different values. We have used here 6th order plynomial for generating a non-linear decision boundary. You can try with different order as well. 

==> The data for model 3 which a digit recognition multiclass classification model with the traning accuracy of 91.46 %. This can be further improved by increasing the number of iteration for the gradient descent algorithm.


****Few important details about logistic regression to know****
- As it's name suggest's that it is regression for predection of continious output value like linear regression , but this is actually a classification algorithm which is souly based on stastitics where we find the relation between the dependent variable and the predectors variables or features.

- Logistic regression uses sigmoid function you can visualize the fuction here https://www.desmos.com/calculator/77rd7fib7e. The principal reason of using this fuction is that if you feed any value in range[-∞,+∞] to this function, it's value will always ranges in 0 to 1. And that is what logistic regression gives us i.e. P(Y=1|θ) which is the probablity of an instance which n features given learned parameters θ belongs to positive class i.e. 1.
Function for hypothesis = (1 / 1 + e^(-transpose(θ)*X)) where θ is the vector and X is the matrix having one or more instances for classification.

- Other importat function which as been used for calculating a cost at perticular point for logistic regression is lograthmic fuction, which you can visualize here https://www.desmos.com/calculator/77rd7fib7e. This function have important property i.e. it is an convex function which means that there is only one global minima for this. If we directly use only sigmod function for this then the cost function is highly non-convex with many local minima.

- Regularization is somtheing which very import when we are trying to fint non-linear functions, because this functions try to over-fit the traning data and does not form a good decison boundry, for this we use this technique. 

- For multiclass classification we use one/all method where we transform the traning set one at a time to form dataset having only binar class, So if you have 4 classes to predect then you have to train 4 logistic regression models. 

- To use logistic regression on any problem statment always see if the data set variables are not dependent. Also it highly computationally inappropriate to have non-linear hypothesis with large number of variables. 
