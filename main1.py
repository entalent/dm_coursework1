import os
import sys
import csv
import codecs
from collections import namedtuple, Counter

import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats

import pandas as pd
import json
import knnimpute

plot_save_path = 'plot2'
digest_file_name = 'digest2.json'


def is_empty_data(value: str):
    v = value.lower()
    return len(value) == 0 or 'na' == v or 'none' == v or 'nan' == v


def read_dataset(csv_file, flag_file):
    # f = codecs.open(csv_file, encoding='utf-8-sig') # UTF-8 with BOM
    # f_csv = csv.reader(f)
    # headers = next(f_csv)
    # # print('\t'.join(headers))
    # # for row in f_csv:
    # #     print(row)
    # f.close()

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

    #impute_data_knn(data_numeric)
    impute_data_column(data_numeric)


def impute_data_knn(data: pd.DataFrame, n_nearest_samples=5):
    data_arr = np.array(data)
    data_arr = data_arr[:10000]
    missing_mask = np.zeros(np.shape(data_arr), np.int8)
    for i in range(np.shape(missing_mask)[0]):
        for j in range(np.shape(missing_mask)[1]):
            missing_mask[i][j] = 1 if np.isnan(data_arr[i][j]) else 0

    X_imputed = knnimpute.knn_impute_few_observed(data_arr, missing_mask, k=n_nearest_samples)


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
    print(cor)

    cols_min = np.nanmin(data_arr, axis=0)  # ignore nan
    cols_max = np.nanmax(data_arr, axis=0)
    print(cols_min, cols_max)
    for row_index in range(num_samples):
        if (row_index) % 1000 == 0:
            print(row_index)
        row = data_arr[row_index, :]
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
            weights = corr_vec / np.sum(corr_vec)
            missing_value = np.sum(weights * row) * (col_max - col_min) + col_min
            data_imputed[row_index][col_index] = missing_value
    return data_imputed








def main():
    # read_dataset(u"E:\\BaiduNetdiskDownload\作业1数据集\\NFL Play by Play 2009-2017 (v4).csv\\NFL Play by Play 2009-2017 (v4).csv", '1.csv')
    read_dataset(u"E:\\BaiduNetdiskDownload\\作业1数据集\\Building_Permits.csv\\Building_Permits.csv", '2.csv')
    # read_dataset(os.path.join('dataset', 'NFL Play by Play 2009-2017 (v4).csv'))


if __name__ == '__main__':
    main()