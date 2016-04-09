# -*- coding: utf-8 -*-
"""
Created on Wed Apr 06 19:53:27 2016

@author: fanqi
"""

import dataProcess
import numpy as np
import time
import csv
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# 这个函数好像没什么用
def scoreForest(estimator, X, y):
    score = estimator.oob_score_
    print 'oob_score_:', score
    return score

if __name__ == '__main__':
    #　因为用的是ＲＦ，所以就没有用ＰＣＡ。 没有进行负采样，这是为何？
    input_df, submit_df = dataProcess.getDataSets(bins = True, scaled = True, binary = True)

    submit_ids = submit_df['PassengerId']
    input_df.drop('PassengerId', axis = 1, inplace = 1)
    submit_df.drop('PassengerId', axis = 1, inplace = 1)
    features_list = input_df.columns.values[1::]

    X = input_df.values[:, 1::]
    y = input_df.values[:, 0]
    survived_weight = .75
    y_weights = np.array([survived_weight if s == 0 else 1 for s in y])

    print 'Rough fitting a RandomForest to determine feature importance...'
    forest = RandomForestClassifier(oob_score = True, n_estimators = 10000)
    forest.fit(X, y, sample_weight = y_weights) #　为每个样本设置权重
    feature_importance = forest.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())

    # 画出权重大于18的特征权重图，之所以是18，看上一句。
    fi_threshold = 18
    important_idx = np.where(feature_importance > fi_threshold)[0]
    important_features = features_list[important_idx]
    print '\n', important_features.shape[0], 'Important features(>', fi_threshold, '% of max importance)...\n'
    sorted_idx = np.argsort(feature_importance[important_idx])[::-1] #　逆序，变为从大到小

    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.title('Feature Importance')
    plt.barh(pos, feature_importance[important_idx][sorted_idx[::-1]], color = 'r', align = 'center')
    plt.yticks(pos, important_features[sorted_idx[::-1]])
    plt.xlabel('Relative Importance')
    plt.draw()
    plt.show()

    X = X[:, important_idx][:, sorted_idx] # 对选中的重要特征进行排序？


    submit_df = submit_df.iloc[:, important_idx].iloc[:, sorted_idx]
    print '\nTraining with', X.shape[1], 'feature:\n', submit_df.columns.values

    sqrtfeat = int(np.sqrt(X.shape[1]))

    minsampsplit = int(X.shape[0] * 0.015)
    params_score = { 'n_estimators'      : 10000,
                     'max_features'      : sqrtfeat,
                     'min_samples_split' : minsampsplit }
    params = params_score
    print 'Generating RandomForestClassifier model with parameters: ', params
    forest = RandomForestClassifier(n_jobs = -1, oob_score = True, **params) # 两个星号，什么意思？

    print '\nFitting model 5 times to get mean OOB score using full training data with class weights...'
    test_scores = []

    for i in range(5):
        forest.fit(X, y, sample_weight = y_weights)
        print 'OOB:', forest.oob_score_ #　Score of the training dataset obtained using an out-of-bag estimate.
        test_scores.append(forest.oob_score_)
    oob = ('%.3f' % (np.mean(test_scores))).lstrip('0')
    oob_std = ('%.3f' % (np.std(test_scores))).lstrip('0')
    oob_lower = ('%.3f' % (np.mean(test_scores) - np.std(test_scores))).lstrip('0')
    print 'OOB Mean:', oob, 'and stddev:', oob_std
    print 'Est. correctly identified test examples:', np.mean(test_scores) * X.shape[0]

    print '\nSubmitting predicted labels for', submit_df.shape[0], 'records with class weights...'
    submission = np.asarray(zip(submit_ids, forest.predict(submit_df))).astype(int)

    print 'Survived weight:', survived_weight
    srv_pct = '%.3f' % (submission[:, 1].mean())
    print 'Died/Survived: ', ".3f" % (1-submission[:, 1].mean()), '/', srv_pct


    output = submission[submission[:, 0].argsort()] # 这一条有什么作用？


    name = 'rfc' + str(int(time.time())) + '.csv'
    print 'Generating results file:', name
    predictions_file = open('./' + name, 'wb')
    open_file_object = csv.writer(predictions_file)
    open_file_object.writerow(['PassengerId', 'Survived'])
    open_file_object.writerows(output) #　将结果写入文件

    print 'Done.'




























