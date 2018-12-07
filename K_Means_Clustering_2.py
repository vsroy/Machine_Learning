import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

x = [5,10,15,24,30,85,71,60,55,80]
y = [3,15,12,10,45,70,80,78,52,91]

plt.scatter(x,y)

#Stroring the array as a numpy array
X = np.array([[5,3],
     [10,15],
     [15,12],
     [24,10],
     [30,45],
     [85,70],
     [71,80],
     [60,78],
     [55,52],
     [80,91],])

#Initializing the KMeans object
k_means = KMeans(n_clusters=2)
k_means.fit(X)

#Getting the final centroids and labels
centroids = k_means.cluster_centers_
labels = k_means.labels_

print(centroids)
print(labels)

colors = ["g.","r.","c.","y"]

#Looping through each training point and plotting the co-ordinates
for i in range(len(X)):
    print("Co-ordinate : ", X[i], "Labels : ", labels[i])
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)

#Plotting the final cluster centroids
plt.scatter(centroids[:,0], centroids[:,1], marker="x", s=150, linewidths=5, zorder = 10)
plt.show()




