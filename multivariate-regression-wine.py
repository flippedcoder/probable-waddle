import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# reads header row from the first line of this file
df = pd.read_csv('./wine-quality-data.csv', header=None)

# take a look at the data
print(df.head())

# add a column of 1s for the bias term because it doesn't change the value you multiply it with
df = pd.concat([pd.Series(1, index=df.index, name='00'), df], axis=1)

# check out the data with this new bias column
print(df.head())

# define the independent (input) variables. since there are 5 features we're working with (quality, pH, alcohol, fixed acidity, and total sulfur dioxide)
X = df.drop(columns=5)

# define the dependent (output) variable. this will be the density prediction.
y = df.iloc[:, 6]

# normalize the input data by dividing each column by the the max value of that column. this helps speed the alogorithm to convergence faster and prevents one feature from dominating the others
for i in range(1, len(X.columns)):
    X[i-1] = X[i-1]/np.max(X[i-1])

# check out the normalized data
print(X.head())

# initialize the values for theta. it can be any other number besides 0, that's just what I chose.
theta = np.array([0]*len(X.columns))

# set the number of training data points to the length of the data minus a few data points to test with
m = len(df) - 500

# define the hypothesis function
def hypothesis(theta, X):
    return theta * X

# define the cost function
def calculateCost(X, y, theta):
    y1 = hypothesis(theta, X)
    y1 = np.sum(y1, axis=1)
    
    return (1 / 2 * m) * sum(np.sqrt((y1 - y) ** 2))

# define the gradient descent function
def gradientDescent(X, y, theta, alpha, i):
    J = [] # cost function for each iteration
    k = 0
    while k < i:
        y1 = hypothesis(theta, X)
        y1 = np.sum(y1, axis=1)
        for c in range(1, len(X.columns)):
            theta[c] = theta[c] - alpha * (1 / m) * (sum((y1 - y) * X.iloc[:, c]))
        j = calculateCost(X, y, theta)
        J.append(j)
        k += 1
    return J, j, theta

# get the final cost, a list of costs on each iteration, and the optimized theta array
J, j, theta = gradientDescent(X, y, theta, 0.1, 10000)

# predict the output using the optimized theta
y_hat = hypothesis(theta, X)
y_hat = np.sum(y_hat, axis=1)

print(y_hat)

# optional: if you want to see a plot of the cost covergence, you can do the following. if you don't want the plot you can also remove the matplotlib import
plt.figure()
plt.scatter(x=list(range(0, 10000)), y=j)
plt.show()