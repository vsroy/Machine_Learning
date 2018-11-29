from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split    #This is important for splitting training and testing data
from sklearn import metrics

iris = datasets.load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

classifier = AdaBoostClassifier(n_estimators=50, learning_rate=1)

model = classifier.fit(x_train, y_train)

predictor = model.predict(x_test)
print("Accuracy score : ", metrics.accuracy_score(y_test,predictor))