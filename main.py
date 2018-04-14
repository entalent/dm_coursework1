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

plot_save_path = 'plot'
digest_file_name = 'digest1.json'


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
    data = d[numeric_header_list]
    digest_data = []
    for h in numeric_header_list:
        d = process_numberic_data_column(data, h, process_data)
        digest_data.append(d)

    f_digest_file = open(digest_file_name, 'w')
    json.dump(digest_data, f_digest_file, indent=4)
    f_digest_file.close()


def process_data(arr: np.array):  # ignore missing value
    arr1 = []
    missing_value_count = 0
    for (i, a) in enumerate(arr):
        if not (np.isnan(a) or np.isinf(a)):
            arr1.append(a)
            continue
        missing_value_count += 1
    return np.array(arr1), missing_value_count


def process_data_1(arr: np.array):  # replace missing value by
    arr1 = []
    missing_value_count = 0
    for (i, a) in enumerate(arr):
        if not (np.isnan(a) or np.isinf(a)):
            arr1.append(a)
            continue
        missing_value_count += 1
    return np.array(arr1), missing_value_count


def process_numberic_data_column(data_frame: pd.DataFrame, column_name: str, process_data_func: callable):
    data = np.array(data_frame[column_name])
    data = data.reshape([data.size])
    data, missing_value_count = process_data_func(data)

    digest_data = digest_numeric_data(data, column_name)
    digest_data['missing'] = missing_value_count
    print('\tmissing: {}'.format(missing_value_count))
    plot_numeric_data(data, column_name)
    return digest_data


def digest_numeric_data(data: np.array, column_name: str):
    max_value = np.max(data)
    min_value = np.min(data)
    avg_value = np.average(data)
    median = np.median(data)
    quantile_25, quantile_50, quantile_75 = [np.percentile(data, i) for i in (25, 50, 75)]

    print('column: {}'.format(column_name))
    print('\tmax: {}'.format(max_value))
    print('\tmin: {}'.format(min_value))
    print('\taverage: {}'.format(avg_value))
    print('\tmedian: {}'.format(median))
    print('\tquantile: {}, {}, {}'.format(quantile_25, quantile_50, quantile_75))

    return {'max': float(max_value), 'min': float(min_value), 'avg': float(avg_value), 'median': float(median), 'quantile': [float(i) for i in (quantile_25, quantile_50, quantile_75)]}


def plot_numeric_data(data: np.array, column_name: str):
    max_value = np.max(data)
    min_value = np.min(data)
    plot_steps = 30
    bins = [min_value + i * (float(max_value - min_value) / plot_steps) for i in range(plot_steps + 1)]

    plt.figure(1, figsize=(28, 8), dpi=80)
    # plt.style.use('classic')

    plt.subplot(1, 3, 1)
    # histogram
    # plt.hist(data, bins=bins, )
    plt.hist(data, bins=bins)
    plt.title("histogram - {}".format(column_name))

    plt.subplot(1, 3, 2)
    # quantile-quantile
    stats.probplot(data, dist='norm', plot=plt)
    plt.title('quantile-quantile - {}'.format(column_name))

    plt.subplot(1, 3, 3)
    # box-plot
    plt.boxplot(data, vert=True)
    plt.title("box plot - {}".format(column_name))
    # plt.show()

    plt.savefig(fname=os.path.join(plot_save_path, 'plot - {}.png'.format(column_name)), format='png')
    plt.close()


def process_nominal_data_column(data_frame: pd.DataFrame, column_name: str):
    counter = Counter()
    data = np.array(data_frame[column_name])
    data = data.reshape([data.size])
    missing_value_count = 0
    # for d in data:


def main():
    read_dataset(u"E:\\BaiduNetdiskDownload\作业1数据集\\NFL Play by Play 2009-2017 (v4).csv\\NFL Play by Play 2009-2017 (v4).csv", '1.csv')
    read_dataset(u"E:\\BaiduNetdiskDownload\\作业1数据集\\Building_Permits.csv\\Building_Permits.csv", '2.csv')
    # read_dataset(os.path.join('dataset', 'NFL Play by Play 2009-2017 (v4).csv'))


if __name__ == '__main__':
    main()