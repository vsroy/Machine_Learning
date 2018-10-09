#Implementing a random forest algorithm to classify counterfeit bank notes based on certain features
import pandas as pd
import numpy as np

#Importing the dataset
dataset = pd.read_csv("bill_authentication.csv")

#splitting the feature data and classifier data
X = dataset.iloc[:, 0:4].values
Y = dataset.iloc[:, 4].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)  #splitting training and testing data

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Training the algorithm
from sklearn.ensemble import RandomForestClassifier
regressor = RandomForestClassifier(n_estimators=20, random_state=0)
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)

#Evaluating the algorithm
from sklearn.metrics import confusion_matrix,classification_report, accuracy_score

print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
print(accuracy_score(Y_test, Y_pred))



