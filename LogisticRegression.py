import numpy as np
import matplotlib.pyplot as plt
import random
import time
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
    for i in range(len(x[0, :])):
        x[:, i] = (x[:, i] - np.min(x[:, i])) / (np.max(x[:, i]) - np.min(x[:, i]))
    return x, y


# Define the activation function (logistic/sigmoidal)
def logistic(x, w):
    val= 1 / (1 + np.exp(-(np.dot(w, x.T))))
    return val


# Define heaviside
def hw(x):
    return 1 * (x > 0.5)


def updateWeight(x, y, w, alpha):
    for i in range(len(w)):
       # w[i] = w[i] + alpha * (y - logistic(x[:, i], w[i])) * logistic(x[:, i], w[i]) * (1 - logistic(x[:, i], w[i])) * x[0, i]
        w[i] = w[i] + alpha * (y - logistic(x[0, :], w))* x[0, i]
    return w


# Define the perceptron with logistic activation function and batch upgrade
def batchLogRegression(x, y):
    w = np.ones(np.size(x[0, :])) * 0.001 # initialize w
    t = 0
    y_hat = hw(logistic(x, w))
    missclassific = loss(y_hat, y)

    while missclassific > 2:
        for i in range(len(x[:, 0])):
            alpha = 1000 / (1000 + t)
            t = t + 1
            w = updateWeight(x[i, :], y[i], w, alpha)
        y_hat = hw(logistic(x, w))  # classify all sample points x by using h(w) to get y_hat
        missclassific = loss(y_hat, y)  # calculate loss (y_hat - y)
    return y_hat, w


# Define the perceptron with logistic activation function and stoch upgrade
def stochLogRegression(x, y):
    w = np.ones(np.size(x[0, :])) * 0.001  # initialize w
    t = 0
    # w = [-0.0007755, -0.08305528, 0.08987936]
    # w = [-0.00079216, -0.08511596,  0.09141474]

    y_hat = hw(logistic(x, w))
    missclassific = loss(y_hat, y)

    while missclassific > 0:           # while nbr of missclassified objects are big
        x_shuffle = [[i] for i in range(len(x))]
        random.shuffle(x_shuffle)
        for ind in range(len(x_shuffle)):
            alpha = 1000 / (1000 + t)
            w = updateWeight(x[x_shuffle[ind], :], y[x_shuffle[ind]], w, alpha)
            y_hat = hw(logistic(x, w))  # classify all sample points x by using h(w) to get y_hat
            missclassific = loss(y_hat, y)  # calculate loss (y_hat - y)
            t = t + 1

            print(missclassific)

            # plt.figure(1)
            # yreg = (-w[0]-w[1]*x[:, 1])/w[2]
            # plt.plot(a1, a2, 'ro', label='Data points for English')
            # plt.plot(b2, b2, 'bo', label='Data points for French')
            # plt.plot(x[:, 1], yreg, label='Line')
            # plt.xlabel('x')
            # plt.ylabel('y')
            # plt.title('English and french')
            # plt.legend()
            # plt.show()
            # time.sleep(0.1)

            if missclassific < 0:
                break
    return y_hat, w


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

y_hat, w = stochLogRegression(x, y)  # Run the regression

print("Missclassifications: %d" %loss(y_hat, y))
plt.figure(1)
yreg = (-w[0]-w[1]*x[:, 1])/w[2]
plt.plot(a1, a2, 'ro', label='Data points for English')
plt.plot(b2, b2, 'bo', label='Data points for French')
plt.plot(x[:, 1], yreg, label='Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('English and french')
plt.legend()
plt.show()


plt.figure(2)
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