import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

# the SVM classifier that will be used to train the model
classify = svm.SVC(gamma=0.001, C=100)

# assign the X and y parameters that will be used in the training model
# X is the data
# y is the label of the data
X,y = digits.data[:-10], digits.target[:-10]

# training the model with the SVC
classify.fit(X,y)

# make a prediction using a test data point
print('Classification of third from last data point')
print(classify.predict([digits.data[-9]]))

# evaluate how well the algorithm guessed the value
plt.imshow(digits.images[-9], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()
