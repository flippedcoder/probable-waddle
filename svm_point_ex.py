import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import svm

def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[0], alpha=0.5,
               linestyles=['-'])

# linear data
X = np.array([1, 5, 1.5, 8, 1, 9, 7, 8.7, 2.3, 5.5, 7.7, 6.1])
y = np.array([2, 8, 1.8, 8, 0.6, 11, 10, 9.4, 4, 3, 8.8, 7.5])

# show unclassified data
# plt.scatter(X, y)
# plt.show()

# shaping data for training the model
training_X = np.vstack((X, y)).T
training_y = [0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1]

# define the algorithm for the model
clf = svm.SVC(kernel='linear', C=1.0)

# train the model
clf.fit(training_X, training_y)

w = clf.coef_[0]

a = -w[0] / w[1]

XX = np.linspace(0, 12)
yy = a * XX - clf.intercept_[0] / w[1]

# h0 = plt.plot(XX, yy, 'k-', label='non weighted division')

# plt.scatter(training_X[:, 0], training_X[:, 1], c=training_y)
# plt.legend()
# plt.show()

# non-linear data
circle_X, circle_y = datasets.make_circles(n_samples=300, noise=0.05)

# show unclassified non-linear data
plt.scatter(circle_X[:, 0], circle_X[:, 1], c=circle_y, marker='.')
plt.show()

# make non-linear algorithm for model
nonlinear_clf = svm.SVC(kernel='rbf', C=1.0)

# training non-linear model
nonlinear_clf.fit(circle_X, circle_y)

plt.scatter(circle_X[:, 0], circle_X[:, 1], c=circle_y, s=50, cmap='autumn')
plot_svc_decision_function(nonlinear_clf)
plt.scatter(nonlinear_clf.support_vectors_[:, 0], nonlinear_clf.support_vectors_[:, 1], s=200, lw=1, facecolors='none')
plt.show()