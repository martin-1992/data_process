#!/usr/bin/env python
# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from collections import defaultdict
from sklearn.preprocessing import StandardScaler


def loadData():
    file_path = r'D:\GitHub\data_process\raw_data/'
    train_file_name = file_path + 'train.json'
    test_file_name = file_path + 'test.json'
    # 读取文件
    train_data = pd.read_json(train_file_name).reset_index()
    test_data = pd.read_json(test_file_name).reset_index()

    # 合并文件
    data = pd.concat([train_data, test_data], axis=0)
    train_idx = data.loc[data['interest_level'].notnull()].index
    test_idx = data.loc[data['interest_level'].isnull()].index

    # y值字典映射
    target_num_map = {'high': 0, 'medium':1, 'low': 2}
    y_train = data.loc[data['interest_level'].notnull(), 'interest_level']
    y_train = y_train.map(target_num_map)

    # 高基数特征
    high_cardinality_feats = ['display_address', 'manager_id', 'building_id', 'street_address']
    data = data[high_cardinality_feats]
    for feat in data.columns:
        if data[feat].dtype == 'object':
            # 统计该列中每个值出现的次数，作为每个ID的值
            value_cnt = data[feat].value_counts()
            data[feat] = data[feat].map(value_cnt)

    # train_idx和test_idx都为原始自然索引，所以使用iloc来计数
    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]

    stda = StandardScaler()
    stda_train = stda.fit_transform(train_data)
    stda_test = stda.transform(test_data)

    return stda_train, y_train, stda_test

# 给定一个分类变量创建目标变量的平均值
np.array(X), np.array(y), np.array(Xt)
#create average value of the target variabe given a categorical feature
def convert_dataset_to_avg(X_train, y_train, X_test, rounding=2, cols=None):
    X_train = X_train.tolist()
    y_train = y_train.tolist()
    X_test = X_test.tolist()
    # 如列没有指定，则使用全部的列
    if cols == None:
        cols = [k for k in range(0, len(X_test[0]))]
    # 初始化测试集的woe_train
    woe = [[0.0 for i in range(0, len(cols))] for j in range(0, len(X_test))]
    # 统计某列的特征值所对应的target值的和
    goods = []
    # 统计某列的特征值所对应的target值的频数
    bads = []
    # 初始goods和bads字典
    for col in cols:
        goods_dict = defaultdict(int)
        bads_dict = defaultdict(int)
        goods.append(goods_dict)
        bads.append(bads_dict)
    total_cnt = 0.0
    total_sum = 0.0

    for row_idx in range(0, len(X_train)):
        target = y_train[row_idx]
        # total_sum / total_cnt为先验值，当在训练集中计算得到的关于某列的特征值的后验值，
        # 而在测试集中没有该特征值的后验值时，使用先验值
        total_sum += target
        total_cnt += 1.0
        for col_idx in range(0, len(cols)):
            col = cols[col_idx]
            # goods为计算某列的特征值对应的target的和，bads为计算某列的特征值对应的target的频数
            # 如a列有个特征值为'aa'，有3个target值为：0,1,1, 则其goods为(0+1+1)=2，bads为(1+1+1)=3
            # 后验概率为2/3=0.66
            goods[col_idx][round(X_train[row_idx][col], rounding)] += target
            bads[col_idx][round(X_train[row_idx][col], rounding)] += 1.0

    for row_idx in range(0, len(X_test)):
        for col_idx in range(0, len(cols)):
            col = cols[col_idx]
            if round(X_test[row_idx][col], rounding) in goods[col_idx]:
                # 训练集得到的特征值的后验值，代入测试集对应的特征值
                woe[row_idx][col_idx] = float(goods[col_idx][round(X_test[row_idx][col], rounding)]) / float(
                                              bads[col_idx][round(X_test[row_idx][col], rounding)])
            else:
                # 当在训练集中计算得到的关于某列的特征值的后验值，而在测试集中没有该特征值的后验值时，使用先验值
                woe[row_idx][col_idx] = round(total_sum / total_cnt)

    return woe


#converts the select categorical features to numerical via creating averages based on the target variable within kfold.
def convert_to_avg(X, y, Xt, seed=1, cvals=5, roundings=2, columns=None):
    # 如没有指定列，则使用全都的列
    if columns == None:
        columns = [k for k in range(0, (X.shape[1]))]
    # array类型转为list类型
    X = X.tolist()
    Xt = Xt.tolist()
    # 初始化矩阵，大小为其数据集X的大小
    woe_train = [[0.0 for i in range(0, len(X[0]))] for j in range(0, len(X))]
    woe_test = [[0.0 for i in range(0, len(Xt[0]))] for j in range(0, len(Xt))]
    # 如使用全部数据进行训练，会产生过拟合，使用交叉验证，将数据进行分层，如将其分成5份，
    # 使用4份进行训练，剩余1份用作
    kfolder = StratifiedKFold(y, n_folds=cvals, shuffle=True, random_state=seed)
    for train_idx, cv_idx in kfolder:
        # 切分训练集和验证集
        X_train, X_cv = np.array(X)[train_idx], np.array(X)[cv_idx]
        y_train = np.array(y)[train_idx]
        # 计算验证集的woe矩阵
        woe_cv = convert_dataset_to_avg(X_train, y_train, X_cv, rounding=roundings, cols=columns)
        X_cv = X_cv.tolist()
        # 使用训练集得到的后验值代入测试集，如5轮交叉验证，可获得5份测试集，拼在一起即为一份完整的测试集
        '''
        # https://github.com/kaz-Anova/StackNet/blob/master/example/twosigma_kaggle/create_files_v1.py
        # kaz-Anova里有这段，但是没用，去掉
        no = 0
        for real_idx in cv_idx:
            for col_idx in range(0, len(X_cv[0])):
                woe_train[real_idx][col_idx] = X_cv[no][col_idx]
            no += 1
        '''
        no = 0
        for real_idx in cv_idx:
            for col_idx in range(0, len(columns)):
                col = columns[col_idx]
                woe_train[real_idx][col] = woe_cv[no][col_idx]
            no += 1
    # 使用全部的训练集得出的后验值，放入测试集中
    woe_final = convert_dataset_to_avg(np.array(X), np.array(y), np.array(Xt), rounding=roundings, cols=columns)
    '''
    for real_idx in range(0, len(Xt)):
        for col_idx in range(0, len(Xt[0])):
            woe_test[real_idx][col_idx] = Xt[real_idx][col_idx]
    '''
    for real_idx in range(0, len(Xt)):
        for col_idx in range(0, len(columns)):
            col = columns[col_idx]
            woe_test[real_idx][col] = woe_final[real_idx][col_idx]

    return np.array(woe_train), np.array(woe_test)



if __name__ == '__main__':
    # 载入数据
    stda_train, y_train, stda_test = loadData()
    # 获得高基数特征的后验值，作为该列的特征值
    woe_train, woe_test = convert_to_avg(stda_train, y_train, stda_test, seed=1, cvals=5, roundings=2)
