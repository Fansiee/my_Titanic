# -*- coding: utf-8 -*-
"""
Created on Thu Apr 07 10:02:35 2016

@author: fanqi
"""

import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from sklearn.svm import LinearSVC, SVC
from sklearn.learning_curve import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import make_circles
np.set_printoptions(precision=4, threshold=10000, linewidth=160, edgeitems=999, suppress=True)

def createdata():
    X, y = make_classification(1000, n_features = 20, n_informative = 2, n_redundant = 2, n_classes = 2, random_state = 0)

    df = DataFrame( np.hstack((X, y[:, None])), columns = range(20) + ['class'] )
    return df

def preplot(df):
    _ = sns.pairplot(df[:50], vars = [8, 11, 12, 14, 19], hue = 'class', size = 1.5)
    plt.show()

    plt.figure(figsize = (12, 10))
    _ = sns.corrplot(df, annot = False)
    plt.show()

def plot_learning_curve(estimator, title, X, y, ylim = None, cv = None, train_sizes = np.linspace(0.1, 1.0, 5)):

    '''
    画出data在某模型上的learning curve
    参数解释
    -------------
    estimator：你用的分类器
    title：表格的标题
    X：输入的feature(numpy的array类型)
    y：输入的target vector
    ylim：tuple格式的(ymin， ymax)，设定图像中纵坐标的最低点和最高点
    cv：做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    -------------
    '''

    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv = 5, n_jobs = 1, train_sizes = train_sizes)
    train_scores_mean = np.mean(train_scores, axis = 1)
    train_scores_std = np.std(train_scores, axis = 1)
    test_scores_mean = np.mean(test_scores, axis = 1)
    test_scores_std = np.std(test_scores, axis = 1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha = 0.1, color = 'r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha = 0.1, color = 'g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color = 'r', label = 'Training score')
    plt.plot(train_sizes, test_scores_mean, 'o-', color = 'g', label = 'Cross-validation score')

    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.legend(loc = 'best')
    plt.grid('on')
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    plt.show()

# plot_learning_curve(LinearSVC(C = 10.0), 'LinearSVC(C = 10.0)', X, y, ylim = (0.8, 1.01), train_sizes = np.linspace(0.05, 0.2, 5))
# plot_learning_curve(LinearSVC(C = 10.0), 'LinearSVC(C = 10.0)', X, y, ylim = (0.8, 1.1), train_sizes = np.linspace(0.1, 1.0, 5))
# plot_learning_curve(LinearSVC(C=10.0), "LinearSVC(C=10.0) Features: 11&14", X[:, [11, 14]], y, ylim=(0.8, 1.0), train_sizes=np.linspace(.05, 0.2, 5))
# plot_learning_curve(Pipeline([('fs', SelectKBest(f_classif, k = 2)), ('svc', LinearSVC(C = 10.0))]), \
                    # 'SelectKBest(f_classif, k =2) + LinearSVC(C = 10.0)', X, y, ylim = (0.8, 1.0), train_sizes = np.linspace(0.05, 0.2, 5))
# plot_learning_curve(LinearSVC(C = 0.1), 'LinearSVC(C = 0.1)', X, y, ylim = (0.8, 1.0), train_sizes = np.linspace(0.05, 0.2, 5))

# estm = GridSearchCV(LinearSVC(), param_grid = {'C': [0.001, 0.01, 0.1, 1.0, 10.0]})
# plot_learning_curve(estm, 'LinearSVC(C = AUTO)', X, y, ylim = (0.8, 1.0), train_sizes = np.linspace(0.05, 0.2, 5))
# print 'Chosen parameter on 100 datapoints: %s' % estm.fit(X[:500], y[:500]).best_params_

# plot_learning_curve(LinearSVC(C = 0.1, penalty = 'l1', dual = False), 'LinearSVC(C = 0.1, penalty = "11")', X, y, ylim = (0.8, 1.0), train_sizes = np.linspace(0.05, 0.2, 5))

# estm = LinearSVC(C = 0.1, penalty = 'l1', dual = False)
# estm.fit(X[:450], y[:450])
# print 'Coefficients learned: %s' % estm.coef_
# print 'Non-zero coefficients: %s' % np.nonzero(estm.coef_)[1]

X, y = make_circles(n_samples = 1000, random_state = 2)
# plot_learning_curve(LinearSVC(C = 0.25), 'LinearSVC(C = 0.25)', X, y, ylim = (0.4, 1.0), train_sizes = np.linspace(0.1, 1.0, 5))
df = DataFrame(np.hstack((X, y[:, None])), columns = range(2) + ['class'])
_ = sns.pairplot(df, vars = [0, 1], hue = 'class', size = 3.5)

X_extra = np.hstack((X, X[:, [0]] ** 2 + X[:, [1]] ** 2))
plot_learning_curve(LinearSVC(C = 0.25), 'LinearSVC(C = 0.25) + distance feature', X_extra, y, ylim = (0.5, 1.0), train_sizes = np.linspace(0.1, 1.0, 5))

plot_learning_curve(SVC(C = 2.5, kernel = 'rbf', gamma = 1.0), 'SVC(C = 2.5, kernel = "rbf", gamma = 1.0)', X, y, ylim = (0.5, 1.0), train_sizes = np.linspace(0.1, 1.0, 5))












