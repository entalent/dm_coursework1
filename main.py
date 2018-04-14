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


def is_empty_data(value: str):
    v = str(value).lower()
    return len(v) == 0 or 'na' == v or 'none' == v or 'nan' == v


def read_dataset(csv_file, flag_file, digest_file_numeric, digest_file_nominal, plot_dir):
    f = open(flag_file)
    f_csv = csv.reader(f)
    headers = next(f_csv)
    flags = next(f_csv)

    numeric_header_list = []
    nominal_header_list = []
    for i, (header, flag) in enumerate(zip(headers, flags)):
        if flag == '1':
            numeric_header_list.append(header)
        else:
            nominal_header_list.append(header)

    print(numeric_header_list)

    d = pd.read_csv(csv_file, )
    data_numeric = d[numeric_header_list]
    data_nominal = d[nominal_header_list]

    process_nominal_data(data_frame=data_nominal, digest_file_name=digest_file_nominal)
    plot_and_digest(data_numeric, digest_file_name=digest_file_numeric, plot_dir=plot_dir)



def plot_and_digest(data: pd.DataFrame, plot_dir, digest_file_name):
    def process_numberic_data_column(data_frame: pd.DataFrame, column_name: str):
        data = np.array(data_frame[column_name])
        data = data.reshape([data.size])
        data, missing_value_count = drop_missing_value(data)

        digest_data = digest_numeric_data(data, column_name, missing_value_count)
        plot_numeric_data(data, column_name, plot_dir)
        return digest_data

    digest_results = []
    for h in data.keys():
        d = process_numberic_data_column(data, h)
        digest_results.append(d)

    f_digest_file = open(digest_file_name, 'w')
    json.dump(digest_results, f_digest_file, indent=4)
    f_digest_file.close()


def drop_missing_value(arr: np.array):  # ignore missing value
    arr1 = []
    missing_value_count = 0
    for (i, a) in enumerate(arr):
        if not (np.isnan(a) or np.isinf(a)):
            arr1.append(a)
            continue
        missing_value_count += 1
    return np.array(arr1), missing_value_count


def digest_numeric_data(data: np.array, column_name: str, missing_value_count: int):
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
    print('\tquartile: {}, {}, {}'.format(quantile_25, quantile_50, quantile_75))
    print('\tmissing: {}'.format(missing_value_count))

    return {'column name': column_name,
            'info': {
                'max': float(max_value),
                'min': float(min_value),
                'average': float(avg_value),
                'median': float(median),
                'quartile': [float(i) for i in (quantile_25, quantile_50, quantile_75)],
                'missing': missing_value_count
             }
        }


def plot_numeric_data(data: np.array, column_name: str, save_dir: str):
    max_value = np.max(data)
    min_value = np.min(data)
    plot_steps = 30
    bins = [min_value + i * (float(max_value - min_value) / plot_steps) for i in range(plot_steps + 1)]

    plt.figure(1, figsize=(28, 8), dpi=80)

    plt.subplot(1, 3, 1)
    # histogram
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

    plt.savefig(fname=os.path.join(save_dir, 'plot - {}.png'.format(column_name)), format='png')
    plt.close()


def process_nominal_data(data_frame: pd.DataFrame, digest_file_name: str):
    digest_result = []
    for h in data_frame.keys():
        counter, missing_value_count = process_nominal_data_column(data_frame[h], h)
        values_count = {}
        for (k, v) in counter.items():
            values_count[str(k)] = v
        digest_result.append({
            'column': h,
            'missing': missing_value_count,
            'values': values_count,
        })
    f_digest_file = open(digest_file_name, 'w')
    json.dump(digest_result, f_digest_file, indent=4)
    f_digest_file.close()


def process_nominal_data_column(data_frame: pd.DataFrame, column_name: str):
    counter = Counter()
    data = np.array(data_frame)
    data = data.reshape([data.size])
    missing_value_count = 0
    data1 = []
    for d in data:
        if is_empty_data(d):
            missing_value_count += 1
        else:
            data1.append(d)
    counter.update(data1)
    return counter, missing_value_count


def main():
    read_dataset(u"E:\\BaiduNetdiskDownload\作业1数据集\\NFL Play by Play 2009-2017 (v4).csv\\NFL Play by Play 2009-2017 (v4).csv",
                 '1.csv',
                 'results\\NFL\\digest_NFL_numeric.json',
                 'results\\NFL\\digest_NFL_nominal.json',
                 'results\\NFL\\plots'
                 )
    read_dataset(u"E:\\BaiduNetdiskDownload\\作业1数据集\\Building_Permits.csv\\Building_Permits.csv",
                 '2.csv',
                 'results\\BuildingPermits\\digest_BuildingPermits_numeric.json',
                 'results\\BuildingPermits\\digest_BuildingPermits_nominal.json',
                 'results\\BuildingPermits\\plots'
                 )

if __name__ == '__main__':
    main()