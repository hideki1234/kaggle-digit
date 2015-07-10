#!/usr/bin/env python3
'''
Neural Networks: digit recognition - validation module

@Author: Hideki Ikeda
@Date 7/9/15
'''

import pandas as pd
from sklearn.metrics import accuracy_score
from neural import digit_recognizer, normalizeX

def main():
    # loading training data
    data = pd.read_csv('../input/train.csv')
    X_tr = data.values[:, 1:].astype(float)
    X_tr = normalizeX(X_tr)
    y_tr = data.values[:, 0]

    # training neural network
    num_hidden = 25
    lambda_ = 1     # regularization parameter
    maxiter = 50    # max number of iterations
    print('Training: size of hidden layer={}, lambda={}, maximum iterations={}'
            .format(num_hidden, lambda_, maxiter))
    recognizer = digit_recognizer(lambda_=lambda_, num_hidden=num_hidden, maxiter=maxiter)
    recognizer.fit(X_tr, y_tr)

    y_val = recognizer.predict(X_tr)
    print('Accuracy={}\n'.format(accuracy_score(y_tr, y_val)))

    # training neural network
    num_hidden = 25
    lambda_ = 1
    maxiter = 500
    print('Training: size of hidden layer={}, lambda={}, maximum iterations={}'
            .format(num_hidden, lambda_, maxiter))
    recognizer = digit_recognizer(lambda_=lambda_, num_hidden=num_hidden, maxiter=maxiter)
    recognizer.fit(X_tr, y_tr)

    y_val = recognizer.predict(X_tr)
    print('Accuracy={}'.format(accuracy_score(y_tr, y_val)))


if __name__ == '__main__':
    main()
