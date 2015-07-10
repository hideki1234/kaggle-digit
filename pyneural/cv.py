#!/usr/bin/env python3
'''
Neural Networks: digit recognition

@Author: Hideki Ikeda
@Date 7/9/15
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.learning_curve import validation_curve
from neural import digit_recognizer, normalizeX


def main():
    # loading training data
    data = pd.read_csv('../input/train.csv')
    X_tr = data.values[:, 1:].astype(float)
    X_tr = normalizeX(X_tr)
    y_tr = data.values[:, 0]

    param_range = np.array(
            [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    train_scores, test_scores = validation_curve(
            digit_recognizer(maxiter=100), X_tr, y_tr,
            param_name='lambda_', param_range=param_range,
            cv=3, scoring='accuracy', n_jobs=2)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with Neural Network")
    plt.xlabel("$\lambda$")
    plt.ylabel("Score")
    plt.ylim(0.85, 1.05)
    plt.semilogx(param_range, train_scores_mean, label="Training score", color="r")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="g")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.legend(loc="best")
    plt.show()


if __name__ == '__main__':
    main()
