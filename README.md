# Logistic-regression-classifier for digit recogniton
This contains 3 logistic regression models which are as follows
1) linear logistic regression model for binary classification, where we predict whether a student will get admited to a premier college based on the marks obtained in tow subjects.
2) Non-linear logistic regression model for binary classification. In this we predect a manufactured chip is defective on not defective.
3) A multiclass logistic regression classifier for digit recognition.

==> The data of model 1 can be found in ex2data1.txt, where we can visualize the data on the 2-D plot using the python script Plot_data.py. The data seems to be linearly seprable, So we fit a linear hypothesis function to the data. You can try this yourself that, we can fit a non-linear hypothesis to this as well but keep in mind that it should not overfit the data. So, for this overfitting issue or underfitting issue can be solved using regularizaion.


==> The data of model 2 can be found in ex2data2.txt, where this data can be visulalized using the 2-D plot using plot_data_regularized. The learned decision boundary is plotted using contour plots which are just the 2-D visulization of a 3-D function at different values. We have used here 6th order plynomial for generating a non-linear decision boundary. You can try with different order as well. 

==> The data for model 3 which a digit recognition multiclass classification model with the traning accuracy of 91.46 %. This can be further improved by increasing the number of iteration for the gradient descent algorithm.
