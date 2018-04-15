import os
import sys
import csv
import codecs
from collections import namedtuple, Counter

import numpy as np
import time
from matplotlib import pyplot as plt
import scipy.stats as stats

import pandas as pd
import json
import knnimpute

from skmice import *

from main import *

# from scipy.stats import mode


def is_empty_data(value: str):
    v = value.lower()
    return len(value) == 0 or 'na' == v or 'none' == v or 'nan' == v


def read_dataset(csv_file, flag_file, dataset_name, impute_strategy):
    assert(impute_strategy in ['knn', 'column', 'mice', 'mode'])

    f = open(flag_file)
    f_csv = csv.reader(f)
    headers = next(f_csv)
    flags = next(f_csv)

    numeric_header_list = []
    for i, (header, flag) in enumerate(zip(headers, flags)):
        if flag == '1':
            numeric_header_list.append(header)

    print(numeric_header_list)
    d = pd.read_csv(csv_file, )
    data_numeric = d[numeric_header_list]   # 所有数值属性

    col = np.array(data_numeric[['Estimated Cost']])
    sum = 0
    for i in col:
        if np.isnan(i):
            continue
        if i < 60:
            sum += 1
    print (sum, len(col))
    print(np.nanmax(col))

    result_dir = os.path.join('results', 'imputed_1', impute_strategy, dataset_name)
    plot_dir = os.path.join(result_dir, 'plots')
    result_file = os.path.join(result_dir, 'digest_{}.json'.format(dataset_name))
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    print('impute strategy:', impute_strategy)
    if impute_strategy == 'knn':
        data_imputed = impute_data_knn(data_numeric)
    elif impute_strategy == 'column':
        data_imputed = impute_data_column(data_numeric)
    elif impute_strategy == 'mice':
        data_imputed = impute_data_mice(data_numeric)
    elif impute_strategy == 'mode':
        data_imputed = impute_data_mode(data_numeric)
    print('processing imputed data')

    plot_and_digest_arr(data_imputed, numeric_header_list, plot_dir, result_file)


def plot_and_digest_arr(data: np.array, col_names: list, plot_dir, digest_file_name):
    def process_numberic_data_column(data, column_name: str):
        data = data.reshape([data.size])
        data, missing_value_count = drop_missing_value(data)

        digest_data = digest_numeric_data(data, column_name, missing_value_count)
        plot_numeric_data(data, column_name, plot_dir)
        return digest_data

    digest_results = []
    for (i, h) in enumerate(col_names):
        d = process_numberic_data_column(data[:, i], h)
        digest_results.append(d)

    f_digest_file = open(digest_file_name, 'w')
    json.dump(digest_results, f_digest_file, indent=4, sort_keys=True)
    f_digest_file.close()


def get_missing_mask(data):
    missing_mask = np.zeros(np.shape(data), np.int8)
    for i in range(np.shape(missing_mask)[0]):
        for j in range(np.shape(missing_mask)[1]):
            missing_mask[i][j] = 1 if np.isnan(data[i][j]) else 0
    return missing_mask


def impute_data_knn(data: pd.DataFrame, n_nearest_samples=5):
    batch_size = 4000
    data_arr = np.array(data)
    data_imputed = np.zeros(shape=np.shape(data_arr), dtype=np.float32)
    n_samples, n_features = np.shape(data_arr)
    for i in range(0, data_arr.shape[0], batch_size):
        print(i)
        data_arr_batch = data_arr[i: min(i + batch_size, n_samples)]
        missing_mask = get_missing_mask(data_arr_batch)
        x_imputed = knnimpute.knn_impute_few_observed(data_arr_batch, missing_mask, k=n_nearest_samples)
        data_imputed[i: min(i + batch_size, n_samples)] = x_imputed
        # print(np.sum(np.isnan(data_arr_batch)), np.sum(np.isnan(x_imputed)))

    return data_imputed


def get_mode(arr):
    arr_appear = Counter()
    arr_appear.update(arr)
    if max(arr_appear.values()) == 1:
        return arr[0]
    else:
        result = list(arr_appear.items())
        result.sort(key=lambda x: x[1], reverse=True)
        print(result[0][0], result[:5])
        return result[0][0]


def impute_data_mode(data: pd.DataFrame):
    data_arr = np.array(data)
    data_imputed = np.copy(data_arr)
    n_samples, n_features = np.shape(data_arr)
    for col_index in range(n_features):
        print(col_index)
        col, _ = drop_missing_value(data_arr[:, col_index])
        m = get_mode(col)
        row_impute_index = np.where(np.isnan(data_arr[:, col_index]))
        data_imputed[row_impute_index, col_index] = m
        assert(not np.isnan(data_imputed[:, col_index]).any())
    return data_imputed


def impute_data_column(data: pd.DataFrame, n_nearest_cols=3):
    def top_n_argmax(a, n):
        return (-a).argsort()[:n]

    data_arr = np.array(data)
    data_imputed = np.copy(data_arr)
    num_samples, num_features = np.shape(data_arr)
    data_series = [pd.Series(data_arr[:, i]) for i in range(data_arr.shape[1])]
    cor = np.zeros(shape=(num_features, num_features), dtype=np.float64)
    for row_index in range(num_features):
        for j in range(num_features):
            cor[row_index][j] = data_series[row_index].corr(data_series[j])

    cols_min = np.nanmin(data_arr, axis=0)  # ignore nan
    cols_max = np.nanmax(data_arr, axis=0)
    for row_index in range(num_samples):
        row = data_arr[row_index, :]
        if np.isnan(row).all():
            continue
        row_normalized = (row - cols_min) / (cols_max - cols_min)
        filled_cols = np.where(np.logical_not(np.isnan(row)))
        missing_cols = np.where(np.isnan(row))

        mask = [0 if np.isnan(a) else 1 for a in row]
        for col_index in missing_cols:
            col_max = cols_max[col_index]
            col_min = cols_min[col_index]
            corr_vec = mask * cor[col_index, :]
            chosen_cols = []
            if len(filled_cols) < n_nearest_cols:
                chosen_cols = filled_cols
            else:
                chosen_cols = top_n_argmax(corr_vec, n=n_nearest_cols)
                corr_vec = [1 if i in chosen_cols else 0 for i in range(len(row))] * cor[col_index, :]
            if len(corr_vec) <= 0:
                continue
            weights = corr_vec / np.sum(corr_vec)
            missing_value = np.sum(weights * row) * (col_max - col_min) + col_min
            # assert(not np.isnan(missing_value).any())
            data_imputed[row_index][col_index] = missing_value
    return data_imputed


def impute_data_mice(data: pd.DataFrame):
    data_arr = np.array(data)

    for i in np.nditer(data_arr):
        if np.isinf(i):
            print(i)

    mask = get_missing_mask(data_arr)
    mice_imputer = MiceImputer()
    data_imputed, specs = mice_imputer.transform(data_arr)
    return data_imputed


def main():
    for s in ['mode', 'column', 'knn', ]:
        read_dataset(u"E:\\BaiduNetdiskDownload\作业1数据集\\NFL Play by Play 2009-2017 (v4).csv\\NFL Play by Play 2009-2017 (v4).csv",
                     '1.csv', "NFL", s)
        read_dataset(u"E:\\BaiduNetdiskDownload\\作业1数据集\\Building_Permits.csv\\Building_Permits.csv",
                     '2.csv', 'BuildingPermits', s)


if __name__ == '__main__':
    main()