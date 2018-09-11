import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

from dataset import read_dataset


def plot_2d(data, specified_color=None):
    '''

    :param data: shape=(n_class, n_sample, 2)
    :return:
    '''
    X = np.empty(shape=(0,), dtype=np.float32)
    Y = np.empty(shape=(0,), dtype=np.float32)
    C = np.empty(shape=(0,), dtype=np.float32)
    colors = ['red', 'green']
    for (i, d) in enumerate(data):
        d = np.array(d)
        if len(d) <= 0:
            continue
        print(type(d), np.shape(d))
        x, y = d[:, 0], d[:, 1]
        print('x', x.shape, 'y', y.shape)
        color = [colors[i]] * len(d)
        X = np.concatenate((X, x))
        Y = np.concatenate((Y, y))
        C = np.concatenate((C, color))
    if specified_color is None:
        plt.scatter(x=X, y=Y, c=C, marker='+')
    else:
        plt.scatter(x=X, y=Y, c=specified_color, marker='+')
    # plt.show()


def visualize(train_data, dim=2, title='', color=None):
    train_id, train_feature, train_label = train_data
    n_labels = np.max(train_label) + 1
    print('n_labels', n_labels)

    pca = PCA(n_components=dim, copy=True, )
    train_feature_ld = pca.fit_transform(X=train_feature)
    scatter_data = [[] for _ in range(n_labels)]
    for i in range(len(train_feature_ld)):
        label = train_label[i][0]
        scatter_data[label].append(train_feature_ld[i])
    scatter_data = np.array(scatter_data)
    if dim == 2:
        plot_2d(scatter_data, color)
        plt.title(title)



def visualize_cls(train_data, test_data, dim=2,):
    train_id, train_feature, train_label = train_data
    test_id, test_feature, test_label = test_data
    n_labels = 2
    print('n_labels', n_labels)

    pca = PCA(n_components=dim, copy=True, )
    pca.fit(X=train_feature)
    train_feature_ld = pca.transform(X=train_feature)
    test_feature_ld = pca.transform(X=test_feature)

    def to_scatter_data(feature, labels):
        scatter_data = [[] for _ in range(n_labels)]
        for i in range(len(feature)):
            label = labels[i][0]
            scatter_data[label].append(feature[i])
        scatter_data = np.array(scatter_data)
        return scatter_data


    plt.subplot(1, 2, 1)
    plot_2d(to_scatter_data(train_feature_ld, train_label))
    plt.title('train')
    plt.subplot(1, 2, 2)
    plot_2d(to_scatter_data(test_feature_ld, test_label))
    plt.title('test - svm')
    plt.savefig('scatter.png')
    plt.show()


def visualize_cls_hist(train_data, test_data, attr_index, attr_name, bins):
    train_id, train_feature, train_label = train_data
    test_id, test_feature, test_label = test_data

    def split_by_label(feature, attr_index, labels):
        print(attr_name, feature[:, attr_index])
        split_data = [[], []]
        for i in range(len(feature)):
            label = labels[i][0]
            split_data[label].append(feature[i][attr_index])
        return split_data

    plt.subplot(1, 2, 1)
    plt.hist(split_by_label(train_feature, attr_index, train_label), label=['not survived', 'survived'], bins=bins)
    plt.title('train - {}'.format(attr_name))
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(split_by_label(test_feature, attr_index, test_label), label=['not survived', 'survived'], bins=bins)
    plt.title('test - {}'.format(attr_name))
    plt.legend()
    plt.savefig('hist_{}.png'.format(attr_name))
    plt.show()