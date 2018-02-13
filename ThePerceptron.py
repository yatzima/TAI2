import numpy as np
import matlabplot.plot

x1, y1 = np.loadtxt('salammbo_a_en', delimiter=',', unpack=True)
x2, y2 = np.loadtxt('salammbo_a_fr', delimiter=',', unpack=True)

# Scales the data to [0, 1]
x1 = (x1 - np.min(x1))/(np.max(x1) - np.min(x1))
y1 = (y1 - np.min(y1))/(np.max(y1) - np.min(y1))
x2 = (x2 - np.min(x2))/(np.max(x2) - np.min(x2))
y2 = (y2 - np.min(y2))/(np.max(y2) - np.min(y2))

alpha = 0.1

# Write a reader function for the LIBSVM format and scale the data in your set. You can write a simplified
# reader that assumes that all the attributes, including zeros, will have an index, i.e. ignore the sparse format.

#def LIBSVMreader(ds):


# Write the perceptron program as explained in pages 723--725 in Russell-Norvig
# and in the slides and run it on your data set.

# Define the threshold function
def h(x, w) : return np.heaviside(w*x, 0)

# Define the Perceptron Learning Rule
def updateWeight(x, y, w):
    w[0] = w[0] + alpha*(y - h(x,w))*x[0]
    w[1] = w[1] + alpha*(y - h(x,w))*x[1]
    return w

# Define the perceptron
def perceptron(x, y, w):
    return 1