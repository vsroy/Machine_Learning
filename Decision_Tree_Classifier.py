#Implementing a Decision Tree classifier using sklearn to classifer to classify whether a person is male/female given their weight, height and weight

from sklearn import tree

sampleClassifer = tree.DecisionTreeClassifier()     #Creating object for decision tree clasifier

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]      #Sample Data for input training

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']               #Sample data for training output

sampleClassifer = sampleClassifer.fit(X, Y)       #Training the model using sample training data
samplePrediction = sampleClassifer.predict([[190, 70, 43]])      #Prediction using the trained model

print(samplePrediction)            #Printing the result of the model