#Implementation of Logistic Regression from scratch in Python
#Generating data
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

np.random.seed(12)
num_observations = 5000

x1 = np.random.multivariate_normal([0, 0], [[1 ,0.75],[0.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, 0.75],[0.75, 1]], num_observations)

simulated_sep_features = np.vstack((x1, x2)).astype(np.float32)
simulated_labels = np.hstack((np.zeros(num_observations), np.ones(num_observations)))

plt.figure(figsize=(12, 8))
plt.scatter(simulated_sep_features[:,0], simulated_sep_features[:, 1], c=simulated_labels, alpha=0.4)

#Calculating the O/P of the sigmoid curve
def sigmoid(scores):
    return(1 / (1+np.exp(-scores)))

#Calculating the log likelihood(Sum of all the training data)
def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    l_l = np.sum(target*scores -np.log(1+np.exp(scores)))
    return l_l

#Building the logsitic regression function
def Logistic_Regression(features, target, num_steps, learning_rate, add_intercept = False):
    if(add_intercept):
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))

    weights = np.zeros(features.shape[1])

    for step in range(num_steps):
        scores=np.dot(features, weights)
        predictions = sigmoid(scores)

        #updating each weight with gradient
        output_error_signal = target - predictions
        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate*gradient

        #print log likelyhood every so often
        if((step % 10000) == 0):
            print(log_likelihood(features, target, weights))

    return weights


weights = Logistic_Regression(simulated_sep_features, simulated_labels, num_steps=300000, learning_rate=5e-5, add_intercept=True)
