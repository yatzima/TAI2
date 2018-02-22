import numpy as np
import matplotlib.pyplot as plt
import random


# For matrices X and Y, find minimum through a walk (iteration) down the surface.
def batchGradDes(x, y, alpha):
    w = [0, 0.5]  # Initial guess
    w = np.array(w)
    q = np.shape(x)[0]  # nbr of rows (nbr of samples)
    grad = dSSE(x, y, w)
    while grad > epsilon:
        sum1 = []
        sum1 = np.array(sum1)
        sum2 = []
        sum2 = np.array(sum2)
        for i in range(len(y)):
            sum1 = np.append(sum1, y[i] - (w[0] + w[1] * x[i, 1]))
            sum2 = np.append(sum2, x[i, 1] * (y[i] - (w[0] + w[1] * x[i, 1])))
        # Update w
        w[0] = w[0] + (alpha / q) * sum(sum1)
        w[1] = w[1] + (alpha / q) * sum(sum2)
        # Calculate the new grad
        grad = dSSE(x, y, w)
    return w


# For a data set ds and weight array w, find the squared loss
def SSE(x, y, w):
    sum = 0
    for i in range(len(y)):
        sum= sum +(y[i] - (w[1] * x[i] + w[0])) ** 2
    return sum


# For matrices X and Y, find minimum through a walk (iteration) down the surface.
def stochGradDes(x, y, alpha):
    w = [0, 0.5]
    w = np.array(w)
    grad = dSSE(x, y, w)
    while grad > epsilon:
        i = random.randint(0, np.shape(x)[0]-1)  # Pick random object
        # Update weights based on this
        w[0] = w[0] + alpha * (y[i] - (w[0]+w[1]*x[i, 1]))
        w[1] = w[1] + alpha * x[i, 1] * (y[i] - (w[0] + w[1] * x[i, 1]))
        grad = dSSE(x, y, w)
    return w


# Value of gradient of loss function
def dSSE(x, y, w):
    w = np.matrix(w)
    w = w.T
    x = np.matrix(x)
    sum1 = []
    sum2 = []
    sum1 = np.array(sum1)
    sum2 = np.array(sum2)
    i = 0
    for value in y:
        sum1 = np.append(sum1, value - (w[0] + w[1] * x[i, 1]))
        sum2 = np.append(sum2, x[i, 1] * (value - (w[0] + w[1] * x[i, 1])))
        i = i + 1
    gradient_w = [-2 * sum(sum1), -2 * sum(sum2)]
    gradval = np.math.sqrt(gradient_w[0]**2 + gradient_w[1]**2)
    return gradval


# Read data
x1, y1 = np.loadtxt('salammbo_a_en', delimiter=',', unpack=True)
x2, y2= np.loadtxt('salammbo_a_fr', delimiter=',', unpack=True)

# Scale arrays
x1_scaled = (x1 - np.min(x1)) / (np.max(x1) - np.min(x1))
y1_scaled = (y1 - np.min(y1)) / (np.max(y1) - np.min(y1))
x2_scaled = (x2 - np.min(x2)) / (np.max(x2) - np.min(x2))
y2_scaled = (y2 - np.min(y2)) / (np.max(y2) - np.min(y2))

# Constants
alpha = 1
epsilon = 0.001

# Add column of ones to X array
a = np.ones(np.size(x1_scaled))
x1_scaled = np.vstack([a, x1_scaled]).T
x2_scaled = np.vstack([a, x2_scaled]).T
y1_scaled = np.vstack([y1_scaled]).T
y2_scaled = np.vstack([y2_scaled]).T

# Do regression
w_e1 = batchGradDes(x1_scaled, y1_scaled, alpha)  # English regression
w_f1 = batchGradDes(x2_scaled, y2_scaled, alpha)  # French regression
w_e2 = stochGradDes(x1_scaled, y1_scaled, alpha)  # English regression
w_f2 = stochGradDes(x2_scaled, y2_scaled, alpha)  # French regression

# Scale back
ke_batch = w_e1[1] * (np.max(y1) - np.min(y1))/(np.max(x1) - np.min(x1))
me_batch = np.min(y1) + (np.max(y1) - np.min(y1))*(w_e1[0] - w_e1[1]*np.min(x1)/(np.max(x1) - np.min(x1)))
kf_batch = w_f1[1] * (np.max(y2) - np.min(y2))/(np.max(x2) - np.min(x2))
mf_batch = np.min(y2) + (np.max(y2) - np.min(y2))*(w_f1[0] - (w_f1[1]*np.min(x2)/(np.max(x2) - np.min(x2))))
ke_stoch = w_e2[1] * (np.max(y1) - np.min(y1))/(np.max(x1) - np.min(x1))
me_stoch = np.min(y1) + (np.max(y1) - np.min(y1))*(w_e2[0] - w_e2[1]*np.min(x1)/(np.max(x1) - np.min(x1)))
kf_stoch = w_f2[1] * (np.max(y2) - np.min(y2))/(np.max(x2) - np.min(x2))
mf_stoch = np.min(y2) + (np.max(y2) - np.min(y2))*(w_f2[0] - w_f2[1]*np.min(x2)/(np.max(x2) - np.min(x2)))

#Print weights
print("English batch. w0 = %f, w1 = %f" % (me_batch, ke_batch))
print("English stoch. w0 = %f, w1 = %f" % (me_stoch, ke_stoch))
print("French batch.  w0 = %f, w1 = %f" % (mf_batch, kf_batch))
print("French stoch.  w0 = %f, w1 = %f" % (mf_stoch, kf_stoch))

#Print loss
sse1 = SSE(x1, y1, [me_batch, ke_batch])
print("Sum of squared errors, english batch: %f" % sse1)
sse1 = SSE(x1, y1, [me_stoch, ke_stoch])
print("Sum of squared errors, english stoch: %f" % sse1)
sse1 = SSE(x2, y2, [mf_batch, kf_batch])
print("Sum of squared errors, french batch: %f" % sse1)
sse1 = SSE(x2, y2, [mf_stoch, kf_stoch])
print("Sum of squared errors, french stoch: %f" % sse1)

#Compute lines
xreg= np.linspace(0, 80000, 50, True)
y_f1 = mf_batch + kf_batch * xreg
y_f2 = mf_stoch + kf_stoch * xreg
y_e1 = me_batch + ke_batch * xreg
y_e2 = me_stoch + ke_stoch * xreg

#Plot
plt.close("all")
plt.figure(5)
plt.plot(xreg, y_e1, label=("$y= %.3f + %.5f * x$" %(me_batch,ke_batch)))
plt.plot(x1, y1, 'ro', label='Points from data set')
plt.xlabel('Number of characters')
plt.ylabel('Number of A')
plt.title('English - batch gradient descent')
plt.legend()
plt.show()

plt.figure(6)
plt.plot(xreg, y_f1, label=("$y= %.3f + %.5f * x$" %(mf_batch,kf_batch)))
plt.plot(x2, y2, 'ro', label='Points from data set')
plt.xlabel('Number of characters')
plt.ylabel('Number of A')
plt.title('French - batch gradient descent')
plt.legend()
plt.show()

plt.figure(7)
plt.plot(xreg, y_e2, label=("$y= %.3f + %.5f * x$" %(me_stoch,ke_stoch)))
plt.plot(x1, y1, 'ro', label='Points from data set')
plt.xlabel('Number of characters')
plt.ylabel('Number of A')
plt.title('English - Stochastic gradient descent')
plt.legend()
plt.show()

plt.figure(8)
plt.plot(xreg, y_f2, label=("$y= %.3f + %.5f * x$" %(mf_stoch,kf_stoch)))
plt.plot(x2, y2, 'ro', label='Points from data set')
plt.xlabel('Number of characters')
plt.ylabel('Number of A')
plt.title('French - Stochastic gradient descent')
plt.legend()
plt.show()