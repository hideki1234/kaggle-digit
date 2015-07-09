#!/usr/bin/env python3
'''
Neural Networks: digit recognition - back-propagation

@Author: Hideki Ikeda
@Date 7/6/15
'''

import random
import numpy as np
import pandas as pd
from scipy.optimize import minimize


#
# global constants
#
num_input = 28 * 28
num_labels = 10

# label to vector conversion table
# 0 -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# 1 -> [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
# ...
# 9 -> [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
label2vec = np.identity(num_labels)


def packMats(*Mats):
    '''
    Pack matrices to an array
    gradient matrices needs to be packed to pass to scipy.optimize.minimize
    '''
    vecs = [T.reshape(T.size) for T in Mats]
    return np.hstack(vecs)


def packedShapes(*Mats):
    '''
    shapes of matrices in the packed array packMats() returns
    '''
    return [T.shape for T in Mats]


def unpackMats(packed, shapes):
    '''
    Restore matices from an array packed by packMats()
    '''
    Mats = list()
    offset = 0
    for r, c in shapes:
        size = r * c
        end = offset + size
        Mats.append(packed[offset:end].reshape((r,c)))
        offset = end

    return Mats


def randInitializeWeights(l_in, l_out):
    '''
    Randomly initialized the weights of a layer with l_in incoming connections
    and l_out outgoing connections.
    Arguments:
        l_in:  # of incoming connections
        l_out: # of outgoing connections
    return:
        l_out * (l_in + 1) ndarray with random values
    '''
    epsilon_init = 0.12
    return (2 * epsilon_init) * np.random.rand(l_out, l_in + 1) - epsilon_init


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def costFunction(packed, shapes, X, y, lambda_):
    '''
    Returns cost and packed gradients
    '''
    Theta1, Theta2 = unpackMats(packed, shapes)
    m = X.shape[0]
    num_labels = Theta2.shape[0]
    Theta1_grad = np.zeros_like(Theta1)
    Theta2_grad = np.zeros_like(Theta2)
    a_0 = np.ones(m)
    
    # input to hidden layer
    Xt = np.vstack((a_0, X.T))
    A2 = sigmoid(Theta1.dot(Xt))

    # hidden to output layers
    A2 = np.vstack((a_0, A2))
    H_theta = sigmoid(Theta2.dot(A2))

    # calculate cost
    Y = [label2vec[y_i] for y_i in y]
    Y = np.array(Y).T
    Tmp = -Y * np.log(H_theta) - (1 - Y) * np.log(1.0 - H_theta)
    J = np.sum(Tmp) + lambda_ / 2.0 * (np.sum((Theta1 * Theta1)[:, 1:]) + np.sum((Theta2 * Theta2)[:, 1:]))
    J /= m

    # backpropagation
    Delta3 = H_theta - Y
    Delta2 = (Theta2.T.dot(Delta3) * A2 * (1 - A2))[1:,:]

    Theta2_grad = Delta3.dot(A2.T)
    Theta2_reg = np.hstack((np.zeros((Theta2.shape[0], 1)), Theta2[:, 1:]))
    Theta2_grad += lambda_ * Theta2_reg

    Theta1_grad = Delta2.dot(Xt.T)
    Theta1_reg = np.hstack((np.zeros((Theta1.shape[0], 1)), Theta1[:, 1:]))
    Theta1_grad += lambda_ * Theta1_reg

    Theta1_grad /= m
    Theta2_grad /= m

    return J, packMats(Theta1_grad, Theta2_grad)


def training(X, y, num_hidden, lambda_, maxiter, initial_Theta1=None, initial_Theta2=None):
    '''
    Training a neural network model with one hidden layer
    Arguments:
        X:          training data
        y:          training label
        lambda_:    regularization parameter
        maxiter:    maximum iterations
        initial_Theta1: initial weight matrix for input layer to hidden layer
                        if None, random initial parameter is generated
        initial_Theta2: initial weight matrix for hidden layer to output layer
                        if None, random initial parameter is generated
    Return:
        a tuple of weight matrices
    '''
    # initializing parameters
    if initial_Theta1 is None:
        initial_Theta1 = randInitializeWeights(num_input, num_hidden)
    if initial_Theta2 is None:
        initial_Theta2 = randInitializeWeights(num_hidden, num_labels)
    initial_packed = packMats(initial_Theta1, initial_Theta2)
    shapes = packedShapes(initial_Theta1, initial_Theta2)

    # now train it
    costFunc = lambda n : costFunction(n, shapes, X, y, lambda_)
    res = minimize(costFunc, initial_packed, jac=True, method='CG',
            options={'maxiter':maxiter, 'disp':True})

    # the result is packed; unpack it before return
    return unpackMats(res.x, shapes)


def predict(Theta1, Theta2, X):
    '''
    Predicts the label of an input given a trained neural network
    Arguments:
        Theta1: Trained weight matrix for input layer to hidden layer
        Theta2: Trained weight matrix for hidden layer to output layer
        X:      data
    Return:
        the predicted label
    '''
    m = X.shape[0]
    num_labels = Theta2.shape[0]
    p = np.empty((m), dtype=int)  # return value
    a_0 = np.array([1])

    for i, a1 in enumerate(X):
        # input to hidden layer
        a1 = np.hstack((a_0, a1)).T
        z2 = Theta1.dot(a1)
        a2 = sigmoid(z2)

        # hidden to output layer
        a2 = np.hstack((a_0, a2)).T
        z3 = Theta2.dot(a2)
        a3 = sigmoid(z3)

        # find the index of the max prediction
        p[i] = np.argmax(a3)

    return p


def normalizeX(X):
    return X / 255.0


def main():
    # loading training data
    data = pd.read_csv('../input/train.csv')
    X_tr = normalizeX(data.values[:, 1:].astype(float))
    y_tr = data.values[:, 0]

    # training neural network
    num_hidden = 25
    lambda_ = 1     # regularization parameter
    maxiter = 50    # max number of iterations
    print('Training: size of hidden layer={}, lambda={}, maximum iterations={}'
            .format(num_hidden, lambda_, maxiter))
    Theta1, Theta2 = training(X_tr, y_tr, num_hidden, lambda_, maxiter)

    # loadint test data
    data = pd.read_csv('../input/test.csv')
    X_test = normalizeX(data.values.astype(float))

    print('Predicting...')
    y_test = predict(Theta1, Theta2, X_test)

    # save the result
    with open('result.csv', 'w') as f_result:
        f_result.write('"ImageId","Label"\n')
        for i, y in enumerate(y_test, 1):
            f_result.write('{},"{}"\n'.format(i,y))


if __name__ == '__main__':
    main()
