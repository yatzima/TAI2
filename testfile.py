import os
os.environ['KERAS_BACKEND']='tensorflow'

import numpy as np
import scipy as sp
import tensorflow as tf
import random as rn

import keras
from keras import backend as K
from keras.layers import Dense, Input
from keras.models import Model
from keras import metrics, regularizers

from sklearn.metrics import *

import matplotlib.pyplot as plt
import matplotlib as mpl

# The size of the plots.
mpl.rcParams['figure.figsize'] = (5,5)

def pipline(inp_dim,
            n_nod,
            act_fun='relu',
            out_act_fun='sigmoid',
            opt_method='Adam',
            cost_fun='binary_crossentropy',
            lr_rate=0.01,
            lambd=0.0):
    lays = [inp_dim] + n_nod

    main_input = Input(shape=(inp_dim,), dtype='float32', name='main_input')

    X = main_input
    for nod in n_nod:
        X = Dense(nod,
                  activation=act_fun,
                  kernel_regularizer=regularizers.l2(lambd))(X)

    output = Dense(1, activation=out_act_fun)(X)

    method = getattr(keras.optimizers, opt_method)

    model = Model(inputs=[main_input], outputs=[output])
    model.compile(optimizer=method(lr=lr_rate),
                  loss=cost_fun)

    return model


def syn1(N):
    """ data(samples, features)"""

    global seed
    print("Frida testar")
    data = np.empty(shape=(N, 2), dtype=np.float32)
    tar = np.empty(shape=(N,), dtype=np.float32)
    N1 = int(N / 2)

    data[:N1, 0] = 4 + np.random.normal(loc=.0, scale=1., size=(N1))
    data[N1:, 0] = -4 + np.random.normal(loc=.0, scale=1., size=(N - N1))
    data[:, 1] = 10 * np.random.normal(loc=.0, scale=1., size=(N))

    data = data / data.std(axis=0)

    # Target
    tar[:N1] = np.ones(shape=(N1,))
    tar[N1:] = np.zeros(shape=(N - N1,))

    # Rotation
    theta = np.radians(30)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])  # rotation matrix
    data = np.dot(data, R)

    return data, tar


def syn2(N):
    """ data(samples, features)"""

    global seed

    data = np.empty(shape=(N, 2), dtype=np.float32)
    tar = np.empty(shape=(N,), dtype=np.float32)
    N1 = int(N / 2)

    # Positive samples
    data[:N1, :] = 0.8 + np.random.normal(loc=.0, scale=1., size=(N1, 2))
    # Negative samples
    data[N1:, :] = -.8 + np.random.normal(loc=.0, scale=1., size=(N - N1, 2))

    # Target
    tar[:N1] = np.ones(shape=(N1,))
    tar[N1:] = np.zeros(shape=(N - N1,))

    return data, tar


def syn3(N):
    """ data(samples, features)"""

    global seed

    data = np.empty(shape=(N, 2), dtype=np.float32)
    tar = np.empty(shape=(N,), dtype=np.float32)
    N1 = int(2 * N / 3)

    # disk
    teta_d = np.random.uniform(0, 2 * np.pi, N1)
    inner, outer = 2, 5
    r2 = np.sqrt(np.random.uniform(inner ** 2, outer ** 2, N1))
    data[:N1, 0], data[:N1, 1] = r2 * np.cos(teta_d), r2 * np.sin(teta_d)

    # circle
    teta_c = np.random.uniform(0, 2 * np.pi, N - N1)
    inner, outer = 0, 3
    r2 = np.sqrt(np.random.uniform(inner ** 2, outer ** 2, N - N1))
    data[N1:, 0], data[N1:, 1] = r2 * np.cos(teta_c), r2 * np.sin(teta_c)

    # Normalization
    # data = data - data.mean(axis=0)/data.std(axis=0)

    tar[:N1] = np.ones(shape=(N1,))
    tar[N1:] = np.zeros(shape=(N - N1,))

    return data, tar


def regr1(N, v=0):
    """ data(samples, features)"""

    global seed

    data = np.empty(shape=(N, 6), dtype=np.float32)

    uni = lambda n: np.random.uniform(0, 1, n)
    norm = lambda n: np.random.normal(0, 1, n)
    noise = lambda n: np.random.normal(0, 1, n)

    for i in range(4):
        data[:, i] = norm(N)
    for j in [4, 5]:
        data[:, j] = uni(N)

    tar = 2 * data[:, 0] + data[:, 1] * data[:, 2] ** 2 + np.exp(data[:, 3]) + \
          5 * data[:, 4] * data[:, 5] + 3 * np.sin(2 * np.pi * data[:, 5])
    std_signal = np.std(tar)
    tar = tar + v * std_signal * noise(N)

    return data, tar

# seed = 0 means random, seed > 0 means fixed
seed = 0
np.random.seed(seed) if seed else None

d,t = syn1(100)
plt.figure(1)
plt.scatter(d[:,0],d[:,1], c=t)

d,t = syn2(100)
plt.figure(2)
plt.scatter(d[:,0],d[:,1], c=t)

d,t = syn3(100)
plt.figure(3)
plt.scatter(d[:,0],d[:,1], c=t)


def stats_class(x=None, y=None, label='Training', modl=None):
    """
    input :
             x = input
             y = output
             label = "Provided text string"
             modl = the model

    output :
             sensitivity = fraction of correctly classified positive cases
             specificity = fraction of correctly classified negative cases
             accuracy = fraction of correctly classified cases
             loss = typically the cross-entropy error
    """

    def binary(y1):
        y1[y1 > .5] = 1.
        y1[y1 <= .5] = 0.
        return y1

    y_pr = modl.predict(x, batch_size=x.shape[0], verbose=0).reshape(y.shape)

    nof_p, tp, nof_n, tn = [np.count_nonzero(k) for k in [y == 1, y_pr[y == 1.] > 0.5, y == 0, y_pr[y == 0.] <= 0.5]]

    sens = tp / nof_p
    spec = tn / nof_n
    acc = (tp + tn) / (len(y))
    loss = modl.evaluate(x, y, batch_size=x.shape[0], verbose=0)

    A = ['Accuracy', 'Sensitivity', 'Specificity', 'Loss']
    B = [acc, sens, spec, loss]

    print('\n#############  STATISTICS for {} Data ##############\n'.format(label))
    for r in zip(A, B):
        print(*r, sep='   ')
    return print('\n#########################################################\n')


def stats_reg(d=None, d_pred=None, label='Training', estimat=None):
    A = ['MSE', 'CorrCoeff']

    pcorr = np.corrcoef(d, d_pred)[1, 0]

    if label.lower() in ['training', 'trn', 'train']:
        mse = estimat.history['loss'][-1]
    else:
        mse = estimat.history['val_loss'][-1]

    B = [mse, pcorr]

    print('\n#############  STATISTICS for {} Data ##############\n'.format(label))
    for r in zip(A, B):
        print(*r, sep='   ')
    return print('\n###########################################################\n')


def decision_b(X=None, Y1=None):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    # grid stepsize
    h = 0.025

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    Z[Z > .5] = 1
    Z[Z <= .5] = 0

    Y_pr = model.predict(X, batch_size=X.shape[0], verbose=0).reshape(Y1.shape)

    Y = np.copy(Y1)
    Y_pr[Y_pr > .5] = 1
    Y_pr[Y_pr <= .5] = 0
    Y[(Y != Y_pr) & (Y == 0)] = 2
    Y[(Y != Y_pr) & (Y == 1)] = 3

    plt.figure()
    # plt.contourf(xx, yy, Z, cmap=plt.cm.PRGn, alpha = .9)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired)

    plt.scatter(X[:, 0][Y == 1], X[:, 1][Y == 1], marker='+', c='k')
    plt.scatter(X[:, 0][Y == 0], X[:, 1][Y == 0], marker='o', c='k')

    plt.scatter(X[:, 0][Y == 3], X[:, 1][Y == 3], marker='+', c='r')
    plt.scatter(X[:, 0][Y == 2], X[:, 1][Y == 2], marker='o', c='r')

    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.tight_layout()


# seed = 0 means random, seed > 0 means fixed
seed = 0
np.random.seed(seed) if seed else None

# Generate training data
x_train, d_train = syn1(100)

# Define the network, cost function and minimization method
INPUT = {'inp_dim': x_train.shape[1],
         'n_nod': [1],  # number of nodes in hidden layer
         'act_fun': 'tanh',  # activation functions for the hidden layer
         'out_act_fun': 'sigmoid',  # output activation function
         'opt_method': 'sgd',  # minimization method
         'cost_fun': 'binary_crossentropy',  # error function
         'lr_rate': 0.1  # learningrate
         }

# Get the model
model = pipline(**INPUT)

# Print a summary of the model
model.summary()

# Train the model
estimator = model.fit(x_train, d_train,
                      epochs=300,  # Number of epochs
                      # validation_data=(x_val, y_val),  # We don't have any validation dataset!
                      batch_size=x_train.shape[0],  # Use batch learning
                      # batch_size=25,
                      verbose=0)

# Call the stats function to print out statistics for the training
stats_class(x_train, d_train, 'Training', model)

# Some plotting
plt.plot(estimator.history['loss'])
plt.title('Model training')
plt.ylabel('training error')
plt.xlabel('epoch')
plt.legend(['train'], loc=0)
plt.show()

# Show the decision boundary
decision_b(x_train, d_train)