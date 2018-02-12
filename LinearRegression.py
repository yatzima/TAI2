import numpy as np
import matplotlib.pyplot as plt

x1, y1 = np.loadtxt('salammbo_a_en', delimiter=',', unpack=True)
x2, y2 = np.loadtxt('salammbo_a_fr', delimiter=',', unpack=True)
alfa = 1
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

gradDes(x1,y1,alfa) #English regression
gradDes(x2,y2,alfa) #French regression

#For a data set ds, we find the minimum through a walk (iteration) down the surface.
def gradDes(x,y, alfa):
    w_initial = [0,0]
    w = np.array(w_initial)
    while(loss(x,y,w)>eps):
        #update w

#For a data set ds and weight array w, find the squared loss
def loss(x, y, w):
    i=0
    for value in y:
        (value-(w[1]*x[i]+w[0]))**2
        i=i+1

def stochGradDes(x,y,alfa)
