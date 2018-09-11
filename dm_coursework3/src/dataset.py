from collections import *

import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize


def fill_with_mode(column):
    def get_mode(arr):
        arr_appear = Counter()
        arr_appear.update(arr)
        if max(arr_appear.values()) == 1:
            return arr[0]
        else:
            result = list(arr_appear.items())
            result.sort(key=lambda x: x[1], reverse=True)
            return result[0][0]

    c1 = filter(lambda x: len(x) > 0, column)
    # c1 = [float(i) for i in c1]
    fill = str(get_mode(c1))
    for i in range(len(column)):
        if len(column[i]) == 0:
            column[i] = fill


def fill_with_mean(column):
    '''
    :param column: shape = (n_row, )
    :return:
    '''
    c1 = filter(lambda x: len(x) > 0, column)
    c1 = [float(i) for i in c1]
    avg = np.nanmean(c1)
    fill = str(avg)
    for i in range(len(column)):
        if len(column[i]) == 0:
            column[i] = fill


def to_feature_vector(row):
    feat = np.zeros(12)
    feat[int(row[0])] = 1    # Pclass 1, 2, 3 => 0, 1, 2
    feat[3 if row[1] == 'male' else 4] = 1  # male => 3  female => 4
    feat[5:9] = [float(i) for i in row[2:6]]   # age, sibsp, parch, fare
    feat[{'C': 9, 'Q': 10, 'S': 11}[row[6]]] = 1    # embarked
    return feat


def column_normalization(column):
    column = np.array(column, dtype=np.float32)
    column = column / np.linalg.norm(column)
    return column


def read_dataset(csv_file):
    d = pd.read_csv(csv_file, dtype=str).fillna('')
    ids = np.array(d[['PassengerId']])
    data = np.array(d[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']])
    # print(np.shape(ids), np.shape(data), len(data))
    if 'Survived' in d.keys():
        train_labels = np.array(d[['Survived']], dtype=np.int32)
    else:
        train_labels = None

    fill_with_mode(data[:, 0])
    fill_with_mode(data[:, 1])
    fill_with_mean(data[:, 2])
    fill_with_mean(data[:, 3])
    fill_with_mean(data[:, 4])
    fill_with_mean(data[:, 5])
    fill_with_mode(data[:, 6])
    data[:, 2] = column_normalization(data[:, 2])
    data[:, 5] = column_normalization(data[:, 5])

    feature = []
    for i in range(len(data)):
        feature.append(to_feature_vector(data[i]))
    feature = np.array(feature)

    return ids, feature, train_labels


def read_dataset_raw(csv_file):
    d = pd.read_csv(csv_file, dtype=str).fillna('')
    ids = np.array(d[['PassengerId']])
    data = np.array(d[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']])
    # print(np.shape(ids), np.shape(data), len(data))
    if 'Survived' in d.keys():
        train_labels = np.array(d[['Survived']], dtype=np.int32)
    else:
        train_labels = None

    fill_with_mode(data[:, 0])
    fill_with_mode(data[:, 1])
    fill_with_mean(data[:, 2])
    fill_with_mean(data[:, 3])
    fill_with_mean(data[:, 4])
    fill_with_mean(data[:, 5])
    fill_with_mode(data[:, 6])

    feature = []
    for i in range(len(data)):
        feature.append(data[i])
    feature = np.array(feature)

    return ids, feature, train_labels