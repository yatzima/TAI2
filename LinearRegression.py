import numpy as np
import matplotlib.pyplot as plt

x1, y1 = np.loadtxt('salammbo_a_en', delimiter=',', unpack=True)
x2, y2 = np.loadtxt('salammbo_a_fr', delimiter=',', unpack=True)

plt.figure(1)
plt.plot(x1,y1, label='Loaded from file!')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(x2,y2, label='Loaded from file!')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interesting Graph\nCheck it out 2')
plt.legend()
plt.show()

w_f = [0,0] #french
w_e = [0,0] #english
wf = np.array(w_f)
we = np.array(w_e)

