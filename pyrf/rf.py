#!/usr/bin/env python3
'''
Neural Networks: digit recognition - back-propagation

@Author: Hideki Ikeda
@Date 7/11/15
'''

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def main():
    # loading training data
    print('Loading training data')
    data = pd.read_csv('../input/train.csv')
    X_tr = data.values[:, 1:].astype(float)
    y_tr = data.values[:, 0]

    print('Start learning...')
    n_trees = 25
    recognizer = RandomForestClassifier(n_trees)
    recognizer.fit(X_tr, y_tr)

    # loading test data
    print('Loading test data')
    data = pd.read_csv('../input/test.csv')
    X_test = data.values.astype(float)

    print('Predicting...')
    y_test = recognizer.predict(X_test)

    # save the result
    with open('result.csv', 'w') as f_result:
        f_result.write('"ImageId","Label"\n')
        for i, y in enumerate(y_test, 1):
            f_result.write('{},"{}"\n'.format(i,y))


if __name__ == '__main__':
    main()
