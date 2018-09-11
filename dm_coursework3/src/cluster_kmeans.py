import os
import sys

import numpy as np
import pandas
import sklearn.cluster, sklearn.mixture
import matplotlib.pyplot as plt

from dataset import *
from visualize import *

visualize_dim = 2


def main():
    train_id, train_feature, train_label = read_dataset(os.path.join('..', 'dataset', 'Titanic', 'train.csv'))
    test_id, test_feature, _ = read_dataset(os.path.join('..', 'dataset', 'Titanic', 'test.csv'))
    n_train_samples = np.shape(train_id)[0]
    n_test_samples = np.shape(test_id)[0]

    all_feature = np.concatenate((train_feature, test_feature), axis=0)

    # cluster_model = sklearn.cluster.KMeans(n_clusters=2).fit(all_feature)
    # cluster_labels = cluster_model.labels_

    cluster_labels = sklearn.cluster.AgglomerativeClustering(n_clusters=2).fit_predict(all_feature)

    cluster_labels = np.reshape(cluster_labels, (np.size(cluster_labels), 1))
    cluster_labels_train, cluster_labels_test = cluster_labels[:n_train_samples], cluster_labels[n_train_samples:]

    # plt.subplot(2, 2, 1)
    # visualize((train_id, train_feature, train_label), visualize_dim, 'train data')
    plt.subplot(1, 2, 1)
    visualize((train_id, train_feature, cluster_labels_train), visualize_dim, 'train cluster')

    test_labels = np.ones(shape=(len(test_id), 1), dtype=np.int32)
    # plt.subplot(2, 2, 3)
    # visualize((test_id, test_feature, test_labels), visualize_dim, 'test data', color='blue')
    plt.subplot(1, 2, 2)
    visualize((test_id, test_feature, cluster_labels_test), visualize_dim, 'test cluster')

    plt.show()

    f_csv = open('submission_cluster.csv', 'w')
    f_csv.write("PassengerId,Survived" + '\n')
    for i in range(n_test_samples):
        result = cluster_labels_test[i][0]
        line = '{},{}\n'.format(test_id[i][0], result)
        f_csv.write(line)
    f_csv.close()


if __name__ == '__main__':
    main()