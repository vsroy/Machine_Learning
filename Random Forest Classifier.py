#Python script to explore manipulations on a data set

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import  train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import  accuracy_score

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#print(train.info())

#print("The raining data has shape ", train.shape)  #printing rows and columns
#print(train.head())   #printing the first 5 rows of the ML train data set

nans_train = train.shape[0] - train.dropna().shape[0]   #shape[0] = number of rows
nans_test = test.shape[0] - test.dropna().shape[0]

#print("%d Missing values in train data" %nans_train)
#print("%d Missing values in test data" %nans_test)

#print(train.isnull().sum())   #printing train data set with columns having missing values

cat = train.select_dtypes(include=['O'])        #Selecting data type char
#print(cat.apply(pd.Series.nunique))             #Applying unique operation on result

#Now going to replace missing values with replacement values
train.workclass.value_counts(sort="True")

train.workclass.fillna('Private', inplace=True)

train.occupation.value_counts(sort="True")
train.occupation.fillna('Prof-specialty',inplace=True)

train['native.country'].value_counts(sort="True")
train['native.country'].fillna('United-States',inplace=True)

#print(train.isnull().sum())

print(train.target.value_counts() / train.shape[0])     #Finding out the proportion

print(pd.crosstab(train.education, train.target, margins=True) / train.shape[0])

for x in train.columns:
    if train[x].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[x].values))
        train[x] = lbl.transform(list(train[x].values))

print(train.target.value_counts())

#Next, build a ramdom forest model

y = train['target']
del train['target']

X = train
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

#Now ,training the random forest classifier
classifier = RandomForestClassifier(n_estimators=500, max_depth=6)
classifier.fit(x_train, y_train)

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini', max_depth=6, max_features='auto',
                       max_leaf_nodes=None, min_impurity_split=1e-07, min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=500, n_jobs=1, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)

classifier.predict(x_test)

#Making prediction and checking model's accuracy
prediction = classifier.predict(x_test)
acc = accuracy_score(np.array(y_test), prediction)
print("Accuracy score is {}" .format(acc))


