# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:30:50 2016

@author: fanqi
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def createdata(filename):
    df = pd.read_csv(filename, header = 0)
    df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int) # 把female变为1，male变为0.01，成绩降低0.9个点
    x = df['Age'].dropna().groupby([df['Gender'], df['Pclass']]).median().unstack().values
    df['AgeFill'] = df['Age']
    for i in [0, 1]:
        for j in [1, 2, 3]:
            df.loc[(df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j), 'AgeFill'] = x[i, j - 1]
    df['AgeFill'] = df['AgeFill'] * 4 #####
    df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
    df['FamilySize'] = (df['SibSp'] + df['Parch']) ** 2#####
    df['Age*Class'] = df['AgeFill'] * df['Pclass'] * 4#####
    df.loc[df['Fare'].isnull(), 'Fare'] = df['Fare'].median()
    df['a'] = df['Pclass'] ** 3 * df['Fare'] / df['AgeFill'] # 1.小幅度提升 3.立方处理，下面的特征乘以2，加大Pclass的权值，特征0.478个点
    df['namelen'] = df.loc[:, 'Name'].apply(lambda x: len(x)) * 2 # 2.较大幅度提升，0.01435
    y = df.dtypes[df.dtypes.map(lambda x: x == 'object')].index.values
    df = df.drop(y, axis = 1)
    df = df.drop(['Age', 'PassengerId'], axis = 1)
    # df = df.dropna() # 因为使用随机森林，所以不需要去掉缺失值。更为重要的是测试集一定不能去掉任意一条数据！否则的话会产生结果的错位，导致结果出现大量错误！尤其是用于提交的结果！
    # 但是，在sklearn中的随机森林不能有缺失值！因为predict这一步中会把输入转换为浮点数，如果有缺失值的话会报错！所以一定要把所有缺失值都填充上！
    final_data = df.values
    return final_data


def predictmodel(train_data, test_data):
    forest = RandomForestClassifier(n_estimators = 100)
    forest = forest.fit(train_data[:, 1:], train_data[:, 0])
    output = forest.predict(test_data)
    return output

def getid(filename):
    df = pd.read_csv(filename, header = 0)
    getid = df.ix[:, 0]
    return getid

def runit(trainfile, testfile):
    train_data = createdata(trainfile)
    test_data = createdata(testfile)
    output = predictmodel(train_data, test_data)
    output = pd.DataFrame(output, columns = ['Survived'])
    # idnum = getid('test.csv') # 会报错，'Series' object is not callable.
    result = pd.concat([getid('test.csv'), output], axis = 1)
    return result




