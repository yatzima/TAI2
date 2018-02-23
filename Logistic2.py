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


def scale(x):
    x_scaled = np.copy(x)
    for i in range(np.shape(x)[1]):
        x_scaled[:, i] = (x[:,i] - np.min(x[:,i])) / (np.max(x[:,i]) - np.min(x[:,i]))
    return x_scaled


# Define the activation function (heaviside)
# Returns classifications for all objects
def hw(x):
    return 1 * (x > 0.5)


# Define the Perceptron Learning Rule
def updateWeight(x_i, y_i, w, alpha):
    #For nbrOfWeights
    x_i = x_i.T
    for i in range(len(w)):
        #w[i] = w[i] + alpha*(y_i - hw(x_i, w)[0, 0])*x_i[0, i]
        w[i] = w[i] + alpha * (y_i - logistic(x_i[:, 0], w)) * logistic(x_i[:, 0], w) * (1 - logistic(x_i[:, 0], w)) * x_i[i, 0]

    return w

def logistic(x, w):
    val= 1 / (1 + np.exp(-(np.dot(w, x))))
    return val


# Define the perceptron with stochastic update
def perceptron(x, y):
    w = np.ones(np.size(x[0, :]))  # initialize w
    y_hat = hw(logistic(x.T, w))
    missclassific = loss(y_hat, y)
    t = 0

    alpha = 1000 / (1000 + t)
    while missclassific > 3:           # while nbr of missclassified objects are big
        i = random.randint(0, len(y)-1)         # pick random object
        w = updateWeight(x[i, :], y[i], w, alpha)      # update weights based on object i
        y_hat = hw(logistic(x.T, w))                       # classify all sample points x by using h(w) to get y_hat
        missclassific = loss(y_hat, y)          # calculate loss (y_hat - y)
        print(missclassific)
        alpha = 1000/(1000+t)
        t = t+1
    return y_hat, w


# Returns the number of missclassified examples using weights w
def loss(y_hat, y):
    sum = 0
    for i in range(len(y)):
        if (float(y[i]) - y_hat[0, i]) != 0:
            sum = sum+1
    return sum


x, y = LIBSVMreader('salammbo_a_copy.txt')
x_scaled = scale(x)
dummy = np.ones(len(x[:,0]))
x_scaled = np.concatenate([np.matrix(dummy), x_scaled.T])
x_scaled = np.transpose(x_scaled)


y_hat, w_hat = perceptron(x_scaled, y)  # Run the perceptron


print("Missclassifications: %d" %loss(y_hat, y))
print(w_hat)
plt.close("all")
plt.figure(1)
redlabel_added = False
bluelabel_added = False
for i in range(len(x[:,0])):
    if y[i] == 0:
        if redlabel_added & (y_hat[0,i] == y[i]):
            plt.plot(x_scaled[i, 1], x_scaled[i,2],'r+')
        elif redlabel_added:
            plt.plot(x_scaled[i, 1], x_scaled[i, 2], 'ro', label ='Missclassified as 1')
        else:
            plt.plot(x_scaled[i, 1], x_scaled[i, 2], 'r+', label='Class 0')
            redlabel_added = True
    elif y[i] == 1:
        if bluelabel_added & (y_hat[0,i] == y[i]):
            plt.plot(x_scaled[i, 1], x_scaled[i,2],'b+')
        elif bluelabel_added:
            plt.plot(x_scaled[i, 1], x_scaled[i, 2], 'bo', label ='Missclassified as 0')
        else:
            plt.plot(x_scaled[i, 1], x_scaled[i, 2], 'b+', label='Class 1')
            bluelabel_added = True
    else:
        plt.plot(x_scaled[i, 1], x_scaled[i, 2], 'g*', label='Unclassified')
yreg =-(w_hat[0] + w_hat[1] * x_scaled[:,1]) / w_hat[2]
plt.plot(x_scaled[:,1], yreg, label='Separation Line')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('English and french')
plt.legend()
plt.show()