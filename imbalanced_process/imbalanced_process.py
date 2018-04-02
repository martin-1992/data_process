#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import ClusterCentroids
from imblearn.ensemble import EasyEnsemble
from imblearn.over_sampling import SMOTE
from collections import Counter


########################################################################
# 读取mat格式的文件
def readFile(file_name):
    # 读取文件
    mat_file = loadmat(file_name)
    # 转为array格式
    X = np.array(mat_file['X'])
    y = np.array(mat_file['y']).reshape(-1)
    # 正负样本比例，label为1占9.61%，label为0占90.39%
    print('label等于0的比例: ', list(y).count(0) / len(y))
    print('label等于1的比例: ', list(y).count(1) / len(y))
    return X, y

########################################################################
# 使用原始数据集建模
def buildModel(clf, X, y, cv_nums=10, is_random=False):
    # 是否打乱数据
    if is_random == True:
        random_lst = list(np.random.randint(0, 1000, 4))
    elif is_random == False:
        random_lst = [0] * 4

    print('----------各种类别不平衡处理方法结果, 为' + str(cv_nums) + '折交叉验证的f1均值----------')
    # 不做处理，使用原始数据集做预测
    print('原始数据集: ', np.mean(cross_val_score(clf, X, y, scoring='f1', cv=cv_nums)))

    ros = RandomOverSampler(random_state=random_lst[0])
    X_oversampled, y_oversampled = ros.fit_sample(X, y)
    # print(sorted(Counter(y_oversampled).items()))
    print('过采样: ', np.mean(cross_val_score(clf, X_oversampled, y_oversampled, scoring='f1', cv=cv_nums)))

    cc = ClusterCentroids(random_state=random_lst[1])
    X_undersampled, y_undersampled = cc.fit_sample(X, y)
    #print(sorted(Counter(y_undersampled).items()))
    print('欠采样: ', np.mean(cross_val_score(clf, X_undersampled, y_undersampled, scoring='f1', cv=cv_nums)))

    sm = SMOTE(random_state=random_lst[2])
    X_smote, y_smote = sm.fit_sample(X, y)
    #print(sorted(Counter(y_smote).items()))
    print('SMOTE: ', np.mean(cross_val_score(clf, X_smote, y_smote, scoring='f1', cv=cv_nums)))

    # 将样本多的类别划分为若干个集合供不同学习器使用，这样对每个学习器来看都进行了欠采样，
    # 但在全局来看却不会丢失重要信息，假设将正样本的类别划分为10份，负样本的类别只有1份，
    # 这样训练10个学习器，每个学习器使用1份正样本和1份负样本，负样本共用
    ee = EasyEnsemble(random_state=random_lst[3], n_subsets=10)
    X_ee, y_ee = ee.fit_sample(X, y)
    # shape=(n_subsets * rows * cols)
    # print(X_ee.shape)
    # print(sorted(Counter(y_ee[0]).items()))

########################################################################
if __name__ == '__main__':
    X, y = readFile('cardio.mat')
    # 使用logistic回归
    lr = LogisticRegression()
    buildModel(lr, X, y, cv_nums=10, is_random=False)

    # 使用随机森林
    rf = RandomForestClassifier(max_depth=4)
    buildModel(rf, X, y, cv_nums=10, is_random=True)

