import numpy as np
import matplotlib.pyplot as plt

x1, y1 = np.loadtxt('salammbo_a_en', delimiter=',', unpack=True)
x2, y2 = np.loadtxt('salammbo_a_fr', delimiter=',', unpack=True)
alpha = 1
epsilon = 0.1

x1 = (x1 - np.min(x1))/(np.max(x1) - np.min(x1))
y1 = (y1 - np.min(y1))/(np.max(y1) - np.min(y1))
x2 = (x2 - np.min(x2))/(np.max(x2) - np.min(x2))
y2 = (y2 - np.min(y2))/(np.max(y2) - np.min(y2))

alpha = 0.1
epsilon = 0.1

plt.figure(1)
plt.plot(x1,y1,'ro', label='English')
plt.xlabel('x')
plt.ylabel('y')
plt.title('English')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(x2,y2,'ro', label='French')
plt.xlabel('x')
plt.ylabel('y')
plt.title('French')
plt.legend()
plt.show()


a = np.ones(np.size(x1))
x1 = np.vstack([a,x1]).T
x2 = np.vstack([a,x2]).T

#For a data set ds, we find the minimum through a walk (iteration) down the surface.
def batchGradDes(x, y, alpha):
    w = [0.5, 0.5]
    w = np.array(w)
    q = np.size(x)
    while(dSSE(x, y ,w) > epsilon):
        #Update w
        w[0] = w[0] + (alpha/q)*sum(y - (x*w))
        w[1] = w[1] + (alpha/q)*sum(y - (x*w))
    return w

#For a data set ds and weight array w, find the squared loss
def loss(x, y, w):
    i=0
    for value in y:
        (value-(w[1]*x[i]+w[0]))**2
        i=i+1


def stochGradDes(x, y, alpha):
    w = [0, 0]


def stochGradDes(x,y,alpha):
    w = [0.5, 0.5]
    w = np.array(w)
    q = 15
    while dSSE(x, y, w) > epsilon:
        w[0] = w[0] + (alpha / q) * sum(y - (w * x))
        w[1] = w[1] + (alpha / q) * sum(y - (w * x))
    return w


def dSSE(x, y, w):
    w = np.matrix(w)
    w = w.T
    x = np.matrix(x)
    #w[0] = -2*np.sum(y - w*x)
    #w[1] = -2*np.sum(y - w*x)
    print(np.shape(w))
    print(w)
    print(np.shape(x*w))
    print(x*w)
    print(np.matmul(x, w))
    loss_w = [-2*np.sum(y - x*w)]
    loss_w
    print(np.shape(loss_w))

    return np.math.sqrt(loss_w**2)


batchGradDes(x1,y1,alpha) #English regression
batchGradDes(x2,y2,alpha) #French regression
stochGradDes(x1,y1,alpha) #English regression
stochGradDes(x2,y2,alpha) #French regression