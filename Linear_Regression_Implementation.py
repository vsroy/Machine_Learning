#Data Set source : http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/slr/frames/slr06.html
#Python program to implement Linear Regression Technique on a Swedish auto data set for predicting total payment for number of claims 

import pandas as pd
import math

dataset = pd.DataFrame(pd.read_csv("insurance.csv"))
#print(dataset)

x_list = dataset['X'].tolist()
y_list = dataset['Y'].tolist()

#Printing the mean
mean_x = sum(x_list)/len(x_list)
mean_y = sum(y_list)/len(y_list)

#Printing the variance
variance_x = sum((x-mean_x)**2 for x in x_list)
variance_y = sum((y-mean_y)**2 for y in y_list)

print("X stats : mean(x)=%.3f, variance(x)=%.3f" %(mean_x, variance_x))
print("Y stats : mean(y)=%.3f, variance(y)=%.3f" %(mean_y, variance_y))

#Finding the co-variances
covar = 0.0
for i in range(0, len(x_list)):
    covar += (x_list[i] - mean_x)*(y_list[i] - mean_y)
print("Co-Variance = " + str(covar))

B1 = covar/variance_x
B0 = mean_y - (B1*mean_x)

print("Co-efficients B1: %.3f, B0 : %.3f" % (B1, B0))

SSE = 0.0
#Calculating the Root Mean Squared Error
for i in range(0, len(y_list)):
    actual = y_list[i]
    x_val = x_list[i]
    predicted = B1*x_val + B0
    error_squared = (actual - predicted)**2
    SSE += error_squared

mean_error = SSE/len(x_list)
print("Sum of Squared Errors = %.3f" % (math.sqrt(SSE)))
