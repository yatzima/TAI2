import numpy as np
import matplotlib.pyplot as plt

x1, y1 = np.loadtxt('salammbo_a_en', delimiter=',', unpack=True)
x2, y2 = np.loadtxt('salammbo_a_fr', delimiter=',', unpack=True)

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
def gradDes(ds):
    w_f = [0,0] #french
    w_e = [0,0] #english
    wf = np.array(w_f)
    we = np.array(w_e)
