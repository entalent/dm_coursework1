# -*- coding: utf-8 -*-
import gc
import os
import sys
import csv
from bisect import *
import io
import json


from fpgrowth import *

import numpy as np
import pandas as pd


def process_dataset(csv_file, flag_file):
    with open(flag_file) as f:
        reader = csv.reader(f)
        headers = next(reader)[1:]
        flag_isnumeric = next(reader)[1:]
        flag_isenum = next(reader)[1:]
        flag_isboolean = next(reader)[1:]

    numeric_headers, enum_headers, boolean_headers = [], [], []

    for i, (header, flag1, flag2, flag3) in enumerate(zip(headers, flag_isnumeric, flag_isenum, flag_isboolean)):
        if flag1 == '1':
            numeric_headers.append((i, header))
        elif flag2 == '1':
            enum_headers.append((i, header))
        elif flag3 == '1':
            boolean_headers.append((i, header))

    print(len(numeric_headers), numeric_headers)
    print(len(enum_headers), enum_headers)
    print(len(boolean_headers), boolean_headers)

    with io.open(csv_file, encoding='utf-8') as f_csv:
        d = pd.read_csv(f_csv, dtype=str)
        # data_numeric, data_enum, data_boolean = d[[i[1] for i in numeric_headers]], d[[i[1] for i in enum_headers]], \
        #                                         d[[i[1] for i in boolean_headers]]
        data_enum, data_boolean = d[[i[1] for i in enum_headers]], \
                                                d[[i[1] for i in boolean_headers]]
        # data_numeric = process_numeric_data_na(data_numeric, numeric_headers)
        data_enum = process_enum_data(data_enum, enum_headers)
        data_boolean = process_enum_data(data_boolean, boolean_headers)
        # print(data_numeric.shape, data_enum.shape, data_boolean.shape)
        n_samples = data_enum.shape[0]

        processed_dataset = []
        for row_index in range(n_samples):
            # numeric_row = data_numeric[row_index]
            enum_row = data_enum[row_index]
            boolean_row = data_boolean[row_index]
            row = np.concatenate((enum_row, boolean_row), axis=0)
            row = filter(lambda x: x is not None, row)
            processed_dataset.append(row)
    return processed_dataset

item_to_string_dict = {}

def add_item(item, s):
    assert(type(item) == tuple)
    if item not in item_to_string_dict:
        item_to_string_dict[item] = s


def find_bin_index(bins, a):
    return bisect_left(bins, a)


def save_item_to_string_dict():
    global item_to_string_dict
    f_save = open('item_to_string.txt', 'w')
    for (k, v) in item_to_string_dict.items():
        f_save.write('({},{}) {}'.format(k[0], k[1], v))
    # del item_to_string_dict


def process_numeric_data_na(dataframe, header_index_list):
    print(header_index_list)
    processed_data = []
    for (column_index, header) in header_index_list:
        print(header)
        col = np.array(dataframe[header])
        col = np.reshape(col, (col.size,))
        processed_col = np.empty(shape=(col.size,), dtype=object)
        for (i, a) in enumerate(col):
            if (np.isnan(a)):
                processed_col[i] = (column_index, 0)
            else:
                processed_col[i] = (column_index, 1)
        add_item((column_index, 0), '{}:nan'.format(column_index))
        add_item((column_index, 1), '{}:valid'.format(column_index))
        processed_data.append(processed_col)
    return np.array(processed_data).T


def process_numeric_data_discrete(dataframe, header_index_list):
    '''
    使用离散化方法处理数值属性
    :param dataframe: 所有数值属性的列
    :param header_index_list: 所有列的列名称以及列的索引
    :return:
    '''
    def split_bins(column, n_bins=10):
        '''
        将数据划分为20个bin
        :param column:
        :param n_bins:
        :return:
        '''
        c = []
        for i in column:
            if np.isnan(i):
                continue
            c.append(i)
        if len(c) == 0:
            return None
        assert(not np.isnan(c).any())
        min_value, max_value = np.nanmin(c), np.nanmax(c)
        assert(not np.isnan(min_value))
        assert(not np.isnan(max_value))
        bins = [float(max_value - min_value) * i / n_bins + min_value for i in range(n_bins + 1)]
        return bins

    print(header_index_list)
    processed_data = []
    for (column_index, header) in header_index_list:
        print(header)
        col = np.array(dataframe[header])
        col = np.reshape(col, (col.size,))
        bins = split_bins(col)
        processed_col = np.empty(shape=(col.size,), dtype=object)
        if bins is not None:
            for (i, a) in enumerate(col):
                if np.isnan(a):
                    continue
                bin_index = find_bin_index(bins, a)
                processed_col[i] = (column_index, bin_index)
                if bin_index >= len(bins):
                    print(bins, bin_index)
                add_item((column_index, bin_index), '{}:{}'.format(header, bins[bin_index]))
        processed_data.append(processed_col)
    return np.array(processed_data).T


def process_enum_data(dataframe, header_index_list):
    print(header_index_list)
    processed_data = []
    for (column_index, header) in header_index_list:
        print(header)
        col = np.array(dataframe[header])
        col = np.reshape(col, (col.size,))
        value_to_index_dict = {}
        cnt = 0
        for (i, a) in enumerate(col):
            if a is None or len(str(a)) == 0 or (type(a) is float and np.isnan(a)):
                continue
            if str(a).lower() == 'nan':
                print(header, i, a, type(a))
            assert str(a).lower() != 'nan'
            value = str(a)
            if value not in value_to_index_dict:
                value_to_index_dict[value] = cnt
                cnt += 1
        processed_col = np.empty(shape=(col.size,), dtype=object)
        for (i, a) in enumerate(col):
            if a is None or len(str(a)) == 0 or (type(a) is float and np.isnan(a)):
                continue
            value = str(a)
            if value not in value_to_index_dict:
                print(value, column_index, i)
                continue
            processed_col[i] = (column_index, value_to_index_dict[value])
            add_item((column_index, value_to_index_dict[value]), '{}:{}'.format(header, value))
        processed_data.append(processed_col)
    return np.array(processed_data).T


def main():
    csv_file = "/home/mcislab/zwt1/dm_coursework1/dataset/Building_Permits.csv"
    # csv_file = "E:\\BaiduNetdiskDownload\\作业1数据集\\Building_Permits.csv\\Building_Permits.csv"
    flag_file = '2_2.csv'
    processed_dataset = process_dataset(csv_file, flag_file)
    n_samples = len(processed_dataset)

    save_item_to_string_dict()

    frozenDataSet = to_frozen(processed_dataset)
    minSupport = 100
    print('createFPTree')
    fptree, headPointTable = create_fptree(frozenDataSet, minSupport)
    frequentPatterns = {}
    prefix = set([])
    print('mineFPTree')
    mine_fptree(headPointTable, prefix, frequentPatterns, minSupport)
    print("frequent patterns:")

    frequent_list = list(frequentPatterns.items())
    frequent_list.sort(key=lambda x: float(x[1]), reverse=True)
    f_frequent = open('frequent_pattern.json', 'w')
    f_frequent.write('[' + '\n')
    for k, v in frequent_list:
        print(k)
        item = ', '.join([item_to_string_dict[i] for i in k])
        obj = {'item': item, 'support': float(v) / n_samples}
        f_frequent.write('  ' + json.dumps(obj) + ',\n')
    f_frequent.write(']' + '\n')
    f_frequent.close()
    minConf = 0.6

    print('gen association rules')
    rulesGenerator(frequentPatterns, minConf, float(n_samples))
    print("association rules:")

    rules_list = []
    for rule in rules:
        x = rule[0]
        y = rule[1]
        x_str = ', '.join([item_to_string_dict[i] for i in x])
        y_str = ', '.join([item_to_string_dict[i] for i in y])
        support, confidence, lift = rule[2], rule[3], rule[4]
        rules_list.append({"x": x_str, "y": y_str, "sup": support, "conf": confidence, "lift": lift})

    f_rules = open('association_rules.json', 'w')
    f_rules.write('[' + '\n')
    for i in rules_list:
        f_rules.write('  ' + json.dumps(i) + ',\n')
    f_rules.write(']\n')
    f_rules.close()

    f_rules_csv = open('association_rules.csv', 'w')
    f_rules_csv.write(','.join(["\"{}\"".format(i) for i in ["X", "Y", "support", "confidence", "lift"]]) + '\n')
    for rule in rules:
        x = rule[0]
        y = rule[1]
        x_str = ', '.join([item_to_string_dict[i] for i in x])
        y_str = ', '.join([item_to_string_dict[i] for i in y])
        f_rules_csv.write(','.join(["\"{}\"".format(i) for i in [x_str, y_str, rule[2], rule[3], rule[4]]]) + '\n')



if __name__ == '__main__':
    main()