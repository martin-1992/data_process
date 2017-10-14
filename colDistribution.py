#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import random
random.seed(2017)


# col: 统计该列特征值对应的y值的后验 分布，col='manager_id'
# target_col: y列，这里为'interest_level'，因其为三分类，下面设置了三项
# train_df: 训练集，test_df: 测试集，
# nums_cv=5，交叉验证次数
def targetDistribution(col, target_col, train_df, test_df, nums_cv):
    # 原始索引，用于下面使用iloc
    lst_idx = list(range(train_df.shape[0]))
    random.shuffle(lst_idx)

    # 多分类，target值有三类
    #target_num_map = {'high': 0, 'medium':1, 'low': 2}
    target_low = [np.nan] * len(train_df)
    target_medium = [np.nan] * len(train_df)
    target_high = [np.nan] * len(train_df)

    # 使用5折交叉验证，当第一次交叉验证训练集没有测试集需要的后验值时，通过后面几次补全缺失的后验值
    # 即有些特征值只在测试集里，但在训练集没有，当轮到下轮时，这次的测试集作为训练集训练
    for i in range(nums_cv):
        # 初始化字典
        building_level = {}
        for val in train_df[col].values:
            # 初始化特征值的计数字典，计算manager_id列的每个值在y中的分布
            # 即属于第一类有多少个，第二类有多少，第三类有多少等
            building_level[val] = [0, 0, 0]
        # 分成5份，测试集1份，训练集为啥剩下的4份
        test_idx = lst_idx[int((i*train_df.shape[0])/5): int(((i+1)*train_df.shape[0])/5)]
        train_idx = list(set(lst_idx).difference(test_idx))
        for idx in train_idx:
            temp = train_df.iloc[idx]
            # 统计某列的某个特征值的y值分布，如{'xx': [2, 1, 6]}
            if temp[target_col] == 'low':
                building_level[temp[col]][0] += 1
            if temp[target_col] == 'medium':
                building_level[temp[col]][1] += 1
            if temp[target_col] == 'high':
                building_level[temp[col]][2] += 1

        # 遍历训练集的每一行
        # 统计manger_id特征下，interest_level的分布(low、medium、high各有多少)
        for idx in test_idx:
            temp = train_df.iloc[idx]
            # 判断测试集的特征值是否在训练集中存在
            if sum(building_level[temp[col]]) != 0:
                # building_level有三个值，表示某个特征值对应的y值，计算y值为low的占比
                # 即y=low/(y=low+y=medium+y=high)，举例{'xx': [2, 1, 6]}，则y=low的占比为2/(2+1+6)=0.22
                target_low[idx] = building_level[temp[col]][0]* 1.0 / sum(building_level[temp[col]])
                target_medium[idx] = building_level[temp[col]][1]* 1.0 / sum(building_level[temp[col]])
                target_high[idx] = building_level[temp[col]][2]* 1.0 / sum(building_level[temp[col]])

    train_df[col + '_low'] = target_low
    train_df[col + '_medium'] = target_medium
    train_df[col + '_high'] = target_high

    ##########################################################################################
    # 测试集的后验分布
    target_low = []
    target_medium = []
    target_high = []
    building_level = {}

    for val in train_df[col].values:
        # 初始化特征值的计数字典，计算manager_id列的特征值对应的y值频数
        building_level[val] = [0, 0, 0]
    for idx in range(train_df.shape[0]):
        temp = train_df.iloc[idx]
        # 统计某列的某个特征值的y值分布，如{'xx': [2, 1, 6]}
        if temp[target_col] == 'low':
            building_level[temp[col]][0] += 1
        if temp[target_col] == 'medium':
            building_level[temp[col]][1] += 1
        if temp[target_col] == 'high':
            building_level[temp[col]][2] += 1

    for val in test_df[col].values:
        # 判断测试集的特征值是否在训练集中存在
        if val not in building_level.keys():
            target_low.append(np.nan)
            target_medium.append(np.nan)
            target_high.append(np.nan)
        else:
            # building_level有三个值，表示某个特征值对应的y值，计算y值为low的占比
            # 即y=low/(y=low+y=medium+y=high)，举例{'xx': [2, 1, 6]}，则y=low的占比为2/(2+1+6)=0.22
            target_low.append(building_level[val][0]*1.0 / sum(building_level[val]))
            target_medium.append(building_level[val][1]*1.0 / sum(building_level[val]))
            target_high.append(building_level[val][2]*1.0 / sum(building_level[val]))

    test_df[col + '_low'] = target_low
    test_df[col + '_medium'] = target_medium
    test_df[col + '_high'] = target_high

    return train_df, test_df


# col: 统计该列特征值对应的y值的后验 分布，col='manager_id'
# target_col: y列，这里为'interest_level'，因其为三分类，下面设置了三项
# train_df: 训练集，test_df: 测试集
# nums_cv=5，交叉验证次数
# 修改的地方，对交叉验证得出的后验值取平均
def targetDistributionAvg(col, target_col, train_df, test_df, nums_cv):
    # 原始索引，用于下面使用iloc
    lst_idx = list(range(train_df.shape[0]))
    random.shuffle(lst_idx)

    # 多分类，target值有三类
    #target_num_map = {'high': 0, 'medium':1, 'low': 2}
    # 乘以nums_cv，记录每次交叉验证得出的后验值
    target_low = [[0.0] * len(train_df)] * nums_cv
    target_medium = [[0.0] * len(train_df)] * nums_cv
    target_high = [[0.0] * len(train_df)] * nums_cv

    # 使用5折交叉验证，当第一次交叉验证训练集没有测试集需要的后验值时，通过后面几次补全缺失的后验值
    # 即有些特征值只在测试集里，但在训练集没有，当轮到下轮时，这次的测试集作为训练集训练
    for i in range(nums_cv):
        # 初始化字典
        building_level = {}
        for val in train_df[col].values:
            # 初始化特征值的计数字典，计算manager_id列的每个值在y中的分布
            # 即属于第一类有多少个，第二类有多少，第三类有多少等
            building_level[val] = [0, 0, 0]
        # 分成5份，测试集1份，训练集为啥剩下的4份
        test_idx = lst_idx[int((i*train_df.shape[0])/5): int(((i+1)*train_df.shape[0])/5)]
        train_idx = list(set(lst_idx).difference(test_idx))
        for idx in train_idx:
            temp = train_df.iloc[idx]
            # 统计某列的某个特征值的y值分布，如{'xx': [2, 1, 6]}
            if temp[target_col] == 'low':
                building_level[temp[col]][0] += 1
            if temp[target_col] == 'medium':
                building_level[temp[col]][1] += 1
            if temp[target_col] == 'high':
                building_level[temp[col]][2] += 1

        # 遍历训练集的每一行
        # 统计manger_id特征下，interest_level的分布(low、medium、high各有多少)
        for idx in test_idx:
            temp = train_df.iloc[idx]
            # 判断测试集的特征值是否在训练集中存在
            if sum(building_level[temp[col]]) != 0:
                # building_level有三个值，表示某个特征值对应的y值，计算y值为low的占比
                # 即y=low/(y=low+y=medium+y=high)，举例{'xx': [2, 1, 6]}，则y=low的占比为2/(2+1+6)=0.22
                target_low[i][idx] = building_level[temp[col]][0]* 1.0 / sum(building_level[temp[col]])
                target_medium[i][idx] = building_level[temp[col]][1]* 1.0 / sum(building_level[temp[col]])
                target_high[i][idx] = building_level[temp[col]][2]* 1.0 / sum(building_level[temp[col]])

    # 求交叉验证统计的后验值的均值
    train_df[col + '_low'] = np.average(target_low, axis=0)
    train_df[col + '_medium'] = np.average(target_medium, axis=0)
    train_df[col + '_high'] = np.average(target_high, axis=0)

    ##########################################################################################
    # 测试集的后验分布
    target_low = []
    target_medium = []
    target_high = []
    building_level = {}

    for val in train_df[col].values:
        # 初始化特征值的计数字典，计算manager_id列的特征值对应的y值频数
        building_level[val] = [0, 0, 0]
    for idx in range(train_df.shape[0]):
        temp = train_df.iloc[idx]
        # 统计某列的某个特征值的y值分布，如{'xx': [2, 1, 6]}
        if temp[target_col] == 'low':
            building_level[temp[col]][0] += 1
        if temp[target_col] == 'medium':
            building_level[temp[col]][1] += 1
        if temp[target_col] == 'high':
            building_level[temp[col]][2] += 1

    for val in test_df[col].values:
        # 判断测试集的特征值是否在训练集中存在
        if val not in building_level.keys():
            target_low.append(0.0)
            target_medium.append(0.0)
            target_high.append(0.0)
        else:
            # building_level有三个值，表示某个特征值对应的y值，计算y值为low的占比
            # 即y=low/(y=low+y=medium+y=high)，举例{'xx': [2, 1, 6]}，则y=low的占比为2/(2+1+6)=0.22
            target_low.append(building_level[val][0]*1.0 / sum(building_level[val]))
            target_medium.append(building_level[val][1]*1.0 / sum(building_level[val]))
            target_high.append(building_level[val][2]*1.0 / sum(building_level[val]))

    test_df[col + '_low'] = target_low
    test_df[col + '_medium'] = target_medium
    test_df[col + '_high'] = target_high

    return train_df, test_df


if __name__ == '__main__':
    train_df = pd.read_json('D:\PycharmProjects\kaggle\Kaggle-Rental-Listing-Inquireies-master/train.json')
    test_df = pd.read_json('D:\PycharmProjects\kaggle\Kaggle-Rental-Listing-Inquireies-master/test.json')

    # 通过交叉验证得出后验值
    train_df, test_df = targetDistribution('manager_id', 'interest_level', train_df, test_df, 5)
    # 对交叉验证得出的后验值取平均
    train_df, test_df = targetDistributionAvg('manager_id', 'interest_level', train_df, test_df, 5)



