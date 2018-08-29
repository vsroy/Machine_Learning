#Importing all dependencies

import csv
from sklearn.svm import SVR
import numpy as np
import matplotlib.pyplot as plt

plt.switch_backend('agg')

dates = []
prices = []

def GetData(filename):
    with open(filename, 'r') as csvfile:
        csvfilereader = csv.reader(csvfile)
        next(csvfilereader) #Skipping the first row
        for row in csvfilereader:
            dates.append(int(row[0].split('-')[0]))
            prices.append(float(row[1]))
    return

def PredictPrice(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1)) #converting to matrix of N X 1
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)       #Defining the support vector regression models

    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    #Now we will be plotting the points
    plt.scatter(dates, prices, color='black', label='Data') #plotting initial datapoints
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF_Model') # plotting the line made by the RBF kernel
    plt.plot(dates, svr_lin.predict(dates), color='green', label='Linear model')  # plotting the line made by linear kernel
    plt.plot(dates, svr_poly.predict(dates), color='blue', label='Polynomial model')  # plotting the line made by polynomial kernel
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Support Vector Regression")
    plt.legend()
    plt.show()

    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

GetData('AAPL.csv')

predictedPrice = PredictPrice(dates, prices, 29)
