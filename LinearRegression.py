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

#For a data set ds, we find the minimum through a walk (iteration) down the surface.
def batchGradDes(ds, alpha):
    w_f = [0,0] #french
    w_e = [0,0] #english
    wf = np.array(w_f)
    we = np.array(w_e)

#For a data set ds and weight array w, find the squared loss
def loss(x, y, w):
    i=0
    for value in y:
        (value-(w[1]*x[i]+w[0]))**2
        i=i+1