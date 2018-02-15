import numpy as np
import matplotlib.pyplot as plt
#from sklearn import *
#from svmutil import *


a1, a2 = np.loadtxt('salammbo_a_en', delimiter=',', unpack=True)
b1, b2 = np.loadtxt('salammbo_a_fr', delimiter=',', unpack=True)

# Scales the data
a1 = (a1 - np.min(a1)) / (np.max(a1) - np.min(a1))
a2 = (a2 - np.min(a2)) / (np.max(a2) - np.min(a2))
b1 = (b1 - np.min(b1)) / (np.max(b1) - np.min(b1))
b2 = (b2 - np.min(b2)) / (np.max(b2) - np.min(b2))

alpha = 0.1

# Reader function for the LIBSVM format. The
# reader assumes all the attributes, including zeros, will have an index, i.e. ignores the sparse format.
def LIBSVMreader(fileName):
    file = open(fileName, 'r')
    y = []
    x = []
    y = np.array(y)
    x = np.array(x)
    i = 0
    #print(file.read())
    for line in file:
        words = line.split() # vector containing three elements: one classification and two feature values
        y = np.append(y, words[0]) # add classification
        j = 1
        features = []
        features = np.array(features)
        while j < np.size(words):
            value = words[j].split(':')[1]
            value = float(value)
            features = np.append(features, value)
            j = j + 1
        if i==0:
            x = features
        else:
            x = np.vstack([x, features])
        i = i+1
    file.close()

# Write the perceptron program as explained in pages 723--725 in Russell-Norvig
# and in the slides and run it on your data set.


# Define the activation function (heaviside)
def hw1(x,w): return np.heaviside(w*x, 0)


# Define the activation function (logistic/sigmoidal)
def hw2(x, w): return 1 / (1 + np.exp(-(w*x)))


# Define the Perceptron Learning Rule
def updateWeight(x, y, w):
    #For nbrOfWeights
    for i in range(len(w)):
        w[i] = w[i] + alpha*(y[i] - hw1(x[i], w[i]))*x[i]
    return w


# Define the perceptron
def perceptron(x, y):
    w = np.ones(len(x), 1)*0.1
    updateWeight(x, y, w)
    y = hw1(x, w)
    return y


def updadeWeight2(x, y, w):
    for i in range(len(w)):
        w[i] = w[i] + alpha*(y[i] - hw2(x[i], w[i])) * hw2(x[i], w[i])*(1 - hw2(x[i], w[i])) * x[i]
    return w


LIBSVMreader('salammbo_a')
plt.figure(1)
plt.plot(a1, a2,'ro', label='Data points for English')
plt.plot(b2, b2,'bo', label='Data points for French')
plt.xlabel('x')
plt.ylabel('y')
plt.title('English and french')
plt.legend()
plt.show()


# As a stop criterion, you will use the number of misclassified examples. ??