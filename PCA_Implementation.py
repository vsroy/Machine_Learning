#Program to implement PCA on IRIS data set
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


#The iris data set
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

#Load the data set into the pandas data frame
df = pd.read_csv(url, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'target'])

#print(df)
features = ['sepal length', 'sepal width', 'petal length', 'petal width']

#Separating out the features from the labels
x = df.loc[:,features].values

y = df.loc[:,['target']].values

x = StandardScaler().fit_transform(x)
#print(x)

#Projecting the 4D features into 2D feature space
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
pCDataFrame = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
finalDF = pd.concat([pCDataFrame, df[['target']]], axis=1)
#print(finalDF)

#Visualizing the 2D Projection
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('PC 1', fontsize = 15)
ax.set_ylabel('PC 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)

targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']

for target, color in zip(targets,colors):
    indicesToKeep = finalDF['target'] == target
    ax.scatter(finalDF.loc[indicesToKeep, 'principal component 1'],
               finalDF.loc[indicesToKeep, 'principal component 2'],
               c = color,
               s = 50)

ax.legend(targets)
ax.grid()
plt.show()

print(pca.explained_variance_ratio_)
