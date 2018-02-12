import numpy as np
import matplotlib.pyplot as plt

x1, y1 = np.loadtxt('salammbo_a_en', delimiter=',', unpack=True)
x2, y2 = np.loadtxt('salammbo_a_fr', delimiter=',', unpack=True)

x1 = (x1 - np.min(x1))/(np.max(x1) - np.min(x1))
y1 = (y1 - np.min(y1))/(np.max(y1) - np.min(y1))
x2 = (x2 - np.min(x2))/(np.max(x2) - np.min(x2))
y2 = (y2 - np.min(y2))/(np.max(y2) - np.min(y2))

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

x1 =


#For a data set ds, we find the minimum through a walk (iteration) down the surface.
def batchGradDes(x, y, alpha):
    w = [0, 0]
    w = np.array(w)
    q = 15
    while(dSSE(x, y ,w) > 0.1):
        w[0] = w[0] + (alpha/q)*sum(y - (w*x))
        w[1] = w[1] + (alpha/q)*sum(y - (w*x))


#For a data set ds and weight array w, find the squared loss
def loss(x, y, w):
    i=0
    for value in y:
        (value-(w[1]*x[i]+w[0]))**2
        i=i+1

def dSSE(x, y, w):
    w[0] = -2*np.sum(y - w*x)
    w[1] = -2*np.sum(y - w*x)
    return w