import os
import sys
import random

import numpy as np
import pandas as pd
from sklearn import svm

from dataset import *
from visualize import *


def get_meshgrid(data):
    step = 0.3
    data_dim = np.shape(data)[1]
    col_min, col_max = np.min(data, axis=1), np.max(data, axis=1)
    coor = []
    for i in range(data_dim):
        coor.append(np.arange(col_min[i], col_max[i], step))
    grid = np.meshgrid(*coor)
    print(np.shape(grid), len(grid))
    return grid


def main():
    # read dataset
    train_id, train_feature, train_label = read_dataset(os.path.join('..', 'dataset', 'Titanic', 'train.csv'))
    test_id, test_feature, _ = read_dataset(os.path.join('..', 'dataset', 'Titanic', 'test.csv'))
    n_train_samples = np.shape(train_id)[0]
    n_test_samples = np.shape(test_id)[0]

    all_index = list(range(n_train_samples))
    random.shuffle(all_index)
    train_split_size = int(n_train_samples * 0.70)
    train_split_index = all_index[:train_split_size]
    test_split_index = all_index[train_split_size:]
    # split dataset
    _train_id, _train_feature, _train_label = train_id[train_split_index], train_feature[train_split_index], \
                                              train_label[train_split_index]
    _test_id, _test_feature, _test_label = train_id[test_split_index], train_feature[test_split_index], \
                                           train_label[test_split_index]

    print(train_feature.shape, train_label.shape)
    print(test_feature.shape)

    clf = svm.SVC()
    clf.fit(X=_train_feature, y=np.reshape(_train_label, newshape=(np.size(_train_label,))))
    test_result = clf.predict(X=_test_feature)

    _test_label = np.reshape(_test_label, newshape=(np.size(_test_label),))
    cnt = 0.0
    for i in range(len(test_result)):
        test_label = 0 if test_result[i] <= 0.50 else 1
        if test_label == int(_test_label[i]):
            cnt += 1
    print('acc: {}'.format(cnt / np.size(_test_label)))

    clf = svm.SVC(kernel='rbf', C=1)
    clf.fit(X=train_feature, y=np.reshape(train_label, newshape=(np.size(train_label,))))
    test_result = clf.predict(X=test_feature)

    f_csv = open('submission_svm.csv', 'w')
    f_csv.write("PassengerId,Survived" + '\n')
    for i in range(n_test_samples):
        result = test_result[i]
        line = '{},{}\n'.format(test_id[i][0], result)
        f_csv.write(line)
    f_csv.close()

    # visualize
    test_result = np.reshape(test_result, newshape=(len(test_result), 1))
    visualize_cls((train_id, train_feature, train_label), (test_id, test_feature, test_result))



if __name__ == '__main__':
    main()