#Program to use PCA to speed up an ML algorithm
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

mnist = fetch_mldata('MNIST original')
from sklearn.model_selection import train_test_split
train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)

#Standardize the data
scaler = StandardScaler()

#Fit on training set only
scaler.fit(train_img)

#Apply transform to both training set and testing set
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

#Make an instance of the model. Use min number of principal components such that 95% of variance is retained
pca = PCA(0.95)

pca.fit(train_img)
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)

#Now apply logistic regression
logisticReg = LogisticRegression(solver='lbfgs')
logisticReg.fit(train_img, train_lbl)
logisticReg.predict(test_img[0].reshape(1,-1))
logisticReg.predict(test_img[0:10])
logisticReg.score(test_img, test_lbl)

