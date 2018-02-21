import numpy as np
import matplotlib.pyplot as plt
import random
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
# reader assumes all the attributes, including zeros,
# will have an index, i.e. ignores the sparse format.
# Returns classification and feature vectors
def LIBSVMreader(fileName):
    file = open(fileName, 'r')
    y = []
    x = []
    y = np.array(y)
    x = np.array(x)
    i = 0
    for line in file:
        words = line.split()  # vector containing three elements: one classification and two feature values
        y = np.append(y, (int)(words[0]))  # add classification to y vector
        features = []  # create empty feature vector for each line
        features = np.array(features)
        j = 1
        while j < np.size(words):
            value = words[j].split(':')[1]  # take the value after :
            value = float(value)
            features = np.append(features, (int)(value))
            j = j + 1
        if i==0:
            x = features  # only needed for the first line
        else:
            x = np.vstack([x, features])
        i = i+1
    file.close()
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x, y


# Define the activation function (logistic/sigmoidal)
#  Returns classifications for all objects
def logistic(x, w):
    return 1 / (1 + np.exp(-(np.dot(w, x.T))))

# Define heaviside
def hw(x):
    return 1 * (x > 0.5)


# Gradient ascent
def updateWeight(x_i, y_i, w):
    #For nbrOfWeights
    for i in range(len(w)):
        w[i] = w[i]+alpha*x_i[0, i]*(y_i - logistic(x_i, w))
    return w


# Define the perceptron with logistic activation function
def logRegression(x, y):
    w = np.ones(np.size(x[0, :])) * 0.1  # initialize w
    y_hat = hw(logistic(x, w))
    missclassific = loss(y_hat, y)
    while missclassific > 2:           # while nbr of missclassified objects are big
        i = random.randint(0, len(y)-1)         # pick random object
        w = updateWeight2(x[i, :], y[i], w)      # update weights based on object i
        y_hat = hw(logistic(x, w))                # classify all sample points x by using h(w) to get y_hat
        missclassific = loss(y_hat, y)        # calculate loss (y_hat - y)
    return y_hat


def updateWeight2(x, y, w):
    for i in range(len(w)):
        w[i] = w[i] + alpha * (y - logistic(x[0, i], w[i])) * logistic(x[0, i], w[i]) * (1 - logistic(x[0, i], w[i])) * x[0, i]
    print(w)
    return w


# Returns the number of missclassified examples using weights w
def loss(y_hat, y):
    sum = 0
    for i in range(len(y)):
        if (float(y[i]) - y_hat[0, i]) != 0:
            sum = sum+1
    return sum


x, y = LIBSVMreader('salammbo_a_copy.txt')
dummy = np.ones(len(x[:,0]))
x = np.concatenate([np.matrix(dummy), x.T])
x = np.transpose(x)

y_hat = logRegression(x, y)  # Run the perceptron

print("Missclassifications: %d" %loss(y_hat, y))
plt.figure(1)
plt.plot(a1, a2, 'ro', label='Data points for English')
plt.plot(b2, b2, 'bo', label='Data points for French')
plt.xlabel('x')
plt.ylabel('y')
plt.title('English and french')
plt.legend()
plt.show()

plt.figure(1)
redlabel_added = False
bluelabel_added = False
for i in range(len(x[:, 0])):
    if y_hat[0, i] == 0:
        if redlabel_added:
            plt.plot(x[i, 1], x[i,2],'r+')
        else:
            plt.plot(x[i, 1], x[i, 2], 'r+', label='Class 0')
            redlabel_added = True
    elif y_hat[0, i] == 1:
        if bluelabel_added:
            plt.plot(x[i, 1], x[i,2],'b+')
        else:
            plt.plot(x[i, 1], x[i, 2], 'b+', label='Class 1')
            bluelabel_added = True
    else:
        plt.plot(x[i, 1], x[i, 2], 'g*', label='Unclassified')
plt.xlabel('x')
plt.ylabel('y')
plt.title('English and french')
plt.legend()
plt.show()