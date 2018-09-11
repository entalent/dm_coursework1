import os
import sys
import math
import random
from collections import *

import pandas as pd
import numpy as np
import tensorflow as tf

from dataset import *
from visualize import *


Config = namedtuple('Config', 'keep_prob, learning_rate',)


class NN:
    def __init__(self, mode, sess_config, config: Config):
        assert(mode in ['train', 'test'])
        self.mode = mode
        self.feature_dim = 12
        self.output_dim = 2     # classification
        self.keep_prob = config.keep_prob
        self.learning_rate = config.learning_rate

        self.session = tf.Session(config=sess_config)
        self.build_model()
        self.session.run(tf.global_variables_initializer())

    def build_model(self):
        self.input = tf.placeholder(dtype=tf.float32, shape=(None, self.feature_dim))
        self.keep_prob_input = tf.placeholder_with_default(self.keep_prob, shape=())
        self.d1 = tf.layers.dense(inputs=self.input, units=self.feature_dim * 2, activation=tf.nn.relu, name="dense1")
        self.d1_dropout = tf.nn.dropout(self.d1, keep_prob=self.keep_prob_input, name="dropout1")
        self.d2 = tf.layers.dense(inputs=self.d1_dropout, units=self.output_dim, name="dense2")
        self.output = tf.nn.softmax(self.d2, name="softmax")

        self.label_input = tf.placeholder(dtype=tf.int64, shape=(None, self.output_dim))
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label_input, logits=self.d2))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss)

    def train_step(self, train_features, train_labels):
        _, loss = self.session.run([self.optimizer, self.loss],
                           feed_dict={self.input: train_features, self.label_input:train_labels, self.keep_prob_input: self.keep_prob })
        return loss

    def inference_step(self, input_features):
        return self.session.run(self.output, feed_dict={self.input: input_features, self.keep_prob_input: 1.0})

    def save(self):
        saver = tf.train.Saver()
        save_path = saver.save(self.session, save_path='nn_model\\model')
        print('saved at:', save_path)

    def load(self):
        saver = tf.train.Saver()
        saver.restore(self.session, save_path='nn_model\\model')


def train(nn, train_feature, train_label, train_steps=1000):
    n_train_samples = np.shape(train_feature)[0]
    all_index = range(n_train_samples)

    def get_train_data(batch_size=32):
        index = random.sample(all_index, k=32)
        feature = train_feature[index]
        label = np.zeros(shape=(batch_size, 2))
        label_data = train_label[index]
        for i in range(batch_size):
            label[i][int(label_data[i][0])] = 1
            # print(label_data[i], label[i])
        return feature, label

    for global_step in range(train_steps):
        f, l = get_train_data()
        loss = nn.train_step(f, l)
        if loss < 0.010:
            break
        print('step {}, loss = {}'.format(global_step, loss))


def test(nn, test_id, test_feature, gt_labels=None):
    n_test_samples = np.shape(test_id)[0]
    f_csv = open('submission_nn.csv', 'w')
    f_csv.write("PassengerId,Survived" + '\n')
    test_labels = []
    for i in range(n_test_samples):
        index = test_id[i]
        feature = [test_feature[i]]
        result = np.argmax(nn.inference_step(input_features=feature)[0])
        line = '{},{}\n'.format(int(index[0]), result)
        test_labels.append(result)
        f_csv.write(line)
        # print(result)
    f_csv.close()

    if gt_labels is not None:
        cnt = 0.0
        gt_labels = np.reshape(gt_labels, newshape=(gt_labels.size,))
        for i in range(n_test_samples):
            # print(gt_labels[i], test_labels[i], gt_labels[i] == test_labels[i])
            if int(gt_labels[i]) == int(test_labels[i]):
                cnt += 1
        print('acc: {}'.format(cnt / float(n_test_samples)))
    return test_labels


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

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    nn = NN('train', sess_config=sess_config, config=Config(keep_prob=0.7, learning_rate=1e-5))

    # train(nn, _train_feature, _train_label, train_steps=2000)
    # test(nn, _test_id, _test_feature, _test_label)

    train(nn, train_feature, train_label, train_steps=2000)
    test_result = test(nn, test_id, test_feature)
    # visualize((test_id, test_feature, np.reshape(test_result, newshape=(len(test_result), 1))), dim=2,
    #           title='test data - nn')
    # plt.show()
    test_result = np.reshape(test_result, newshape=(len(test_result), 1))
    visualize_cls((train_id, train_feature, train_label), (test_id, test_feature, test_result))

    _, train_feature_raw, _ = read_dataset_raw(os.path.join('..', 'dataset', 'Titanic', 'train.csv'))
    _, test_feature_raw, _ = read_dataset_raw(os.path.join('..', 'dataset', 'Titanic', 'test.csv'))
    col_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    bin_nums = [3, 2, 10, 10, 10, 10, 10]
    for i in range(7):
        visualize_cls_hist((train_id, train_feature_raw, train_label), (test_id, test_feature_raw, test_result), attr_index=i,
                           attr_name=col_names[i], bins=bin_nums[i])

if __name__ == '__main__':
    main()