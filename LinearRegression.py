import numpy as np
import matplotlib.pyplot as plt
import random

# Read data
x1, y1 = np.loadtxt('salammbo_a_en', delimiter=',', unpack=True)
x2, y2 = np.loadtxt('salammbo_a_fr', delimiter=',', unpack=True)

# Scale arrays
x1 = (x1 - np.min(x1)) / (np.max(x1) - np.min(x1))
y1 = (y1 - np.min(y1)) / (np.max(y1) - np.min(y1))
x2 = (x2 - np.min(x2)) / (np.max(x2) - np.min(x2))
y2 = (y2 - np.min(y2)) / (np.max(y2) - np.min(y2))

# Constants
alpha = 5
epsilon = 0.01

# Add column of ones to X array
a = np.ones(np.size(x1))
x1 = np.vstack([a, x1]).T
x2 = np.vstack([a, x2]).T
y1 = np.vstack([y1]).T
y2 = np.vstack([y2]).T

# For a matrices X and Y, find minimum through a walk (iteration) down the surface.
def batchGradDes(x, y, alpha):
    w = [0, 0.5]  # Initial guess
    w = np.array(w)
    q = np.shape(x)[0]  # nbr of rows; nbr of points
    loss = dSSE(x, y, w)
   # minLoss = loss
    #lossdiff = loss - minLoss + 2 * epsilon  # make sure we enter loop at least once
    while loss > epsilon:
        #if loss < minLoss:
         #   minLoss = loss
        # Update w
        sum1 = []
        sum1 = np.array(sum1)
        sum2 = []
        sum2 = np.array(sum2)
        i=0
        for value in y:
            sum1 = np.append(sum1, value - (w[0] + w[1] * x[i, 1]))
            sum2 = np.append(sum2, x[i, 1] * value - (w[0] + w[1] * x[i, 1]))
            i = i+1
        w[0] = w[0] + (alpha / q) * sum(sum1)
        w[1] = w[1] + (alpha / q) * sum(sum2)
        loss = dSSE(x, y, w)
        #lossdiff = loss - minLoss
        print(loss)
    return w


# For a data set ds and weight array w, find the squared loss
def loss(x, y, w):
    i = 0
    for value in y:
        (value - (w[1] * x[i] + w[0])) ** 2
        i = i + 1


def stochGradDes(x, y, alpha):
    w = [0, 0.5]
    w = np.array(w)
    loss = dSSE(x, y, w)
    minLoss = loss
    lossdiff = loss - minLoss + 2*epsilon  # make sure we enter loop at least once
    while lossdiff > epsilon:
        if loss < minLoss:
            minLoss = loss
        # Update w
        i = random.randint(0, np.shape(x)[0]-1)
        w[0] = w[0] + alpha * (y[i] - (w[0]+w[1]*x[i, 1]))
        w[1] = w[1] + alpha * x[i, 1] * (y[i] - (w[0] + w[1] * x[i, 1]))
        loss = dSSE(x, y, w)
        lossdiff = loss - minLoss
        print(lossdiff)
    return w


# Gradient of loss function
def dSSE(x, y, w):
    w = np.matrix(w)
    w = w.T
    x = np.matrix(x)
    sum1 = []
    sum2 = []
    i = 0
    for value in y:
        sum1 = np.append(sum1, value - (w[0] + w[1] * x[i, 1]))
        sum2 = np.append(sum2, x[i, 1] * value - (w[0] + w[1] * x[i, 1]))
        i = i + 1
    loss_w = [2 * sum(sum1), -2 * sum(sum2)]
    sqloss = np.math.sqrt(loss_w[0]**2 + loss_w[1]**2)
    # print(sqloss)
    return sqloss


w_e1 = batchGradDes(x1, y1, alpha)  # English regression
w_f1 = batchGradDes(x2, y2, alpha)  # French regression
w_e2 = stochGradDes(x1, y1, alpha)  # English regression
w_f2 = stochGradDes(x2, y2, alpha)  # French regression

# Print out coefficients
print(w_e1)
print(w_f1)
print(w_e2)
print(w_f2)

xreg = np.linspace(0, 1, 50, True)
y_f1 = w_f1[0] + w_f1[1] * xreg
y_f2 = w_f2[0] + w_f2[1] * xreg
y_e1 = w_e1[0] + w_e1[1] * xreg
y_e2 = w_e2[0] + w_e2[1] * xreg

plt.figure(1)
plt.plot(xreg, y_e1, label='English - batch')
plt.plot((x1[:,1]), y1, 'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.title('English')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(xreg, y_f1, label='French - batch')
plt.plot((x2[:,1]), y2, 'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.title('French')
plt.legend()
plt.show()

plt.figure(3)
plt.plot(xreg, y_e2, label='English - stochastic')
plt.plot((x1[:,1]), y1, 'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.title('English')
plt.legend()
plt.show()

plt.figure(4)
plt.plot(xreg, y_f2, label='French - stochastic')
plt.plot((x2[:,1]), y2, 'ro')
plt.xlabel('x')
plt.ylabel('y')
plt.title('French')
plt.legend()
plt.show()