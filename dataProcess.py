# -*- coding: utf-8 -*-
"""
Created on Tue Apr 05 09:26:21 2016

@author: fanqi
"""



import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import re
import numpy as np
import pandas as pd
import random as rd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

def processCabin():
    global df
    df['Cabin'][df.Cabin.isnull()] = 'U0'
    df['CabinLetter'] = df['Cabin'].map(lambda x: getCabinLetter(x)) # map函数
    df['CabinLetter'] = pd.factorize(df['CabinLetter'])[0] # factorize函数 需要看一下
    # 先factorize，然后再get_dummies。但是好像直接get_dummies也行？效果好像一样啊，好像就是一样。
    # 之所以先factorize，是因为不一定get_dummies，后面这个是可选的，如果不get_dummies，那么久直接用factorize的结果
    if keep_binary:
        cletters = pd.get_dummies(df['CabinLetter']).rename(columns = lambda x: 'CabinLetter_' + str(x))
        df = pd.concat([df, cletters], axis = 1)

    df['CabinNumber'] = df['Cabin'].map(lambda x: getCabinNumber(x)).astype(int) + 1  # 加1是因为'U0',但是不加的话好像也没事
    if keep_scaled:
        scaler = preprocessing.StandardScaler() # 变为0均值，1标准差，不一定在(0,1)中，这两者不同
        df['CabinNumber_scaled'] = scaler.fit_transform(df['CabinNumber']) # fit_tranform函数
        # 完全可以直接用df['CabinNumber_scaled'] = preprocessing.scale(df['CabinNumber'])

def getCabinLetter(cabin):
    match = re.compile('([a-zA-Z]+)').search(cabin)
    if match:
        return match.group()
    else:
        return 'U'

def getCabinNumber(cabin):
    match = re.compile('([0-9]+)').search(cabin)
    if match:
        return match.group()
    else:
        return 0

def processTicket():
    global df

    df['TicketPrefix'] = df['Ticket'].map(lambda x: getTicketPrefix(x.upper())) # 此时结果中带有斜杠
    df['TicketPrefix'] = df['TicketPrefix'].map(lambda x: re.sub('[\.?\/?]', '', x)) # 去掉斜杠
    df['TicketPrefix'] = df['TicketPrefix'].map(lambda x: re.sub('STON', 'SOTON', x)) # 替换字符串，把前面的换成后面的
    df['TicketPrefixId'] = pd.factorize(df['TicketPrefix'])[0]

    if keep_binary:
        prefixes = pd.get_dummies(df['TicketPrefix']).rename(columns = lambda x: 'TicketPrefix_' + str(x))
        df = pd.concat([df, prefixes], axis = 1)

    df.drop(['TicketPrefix'], axis = 1, inplace = True)

    df['TicketNumber'] = df['Ticket'].map(lambda x: getTicketNumber(x))
    df['TicketNumberDigits'] = df['TicketNumber'].map(lambda x: len(x)).astype(np.int)# 票号的长度
    df['TicketNumberStart'] = df['TicketNumber'].map(lambda x: x[0:1]).astype(np.int)# 票号开头的数字

    df['TicketNumber'] = df.TicketNumber.astype(np.int)

    if keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['TicketNumber_scaled'] = scaler.fit_transform(df['TicketNumber'])

def getTicketPrefix(ticket):
    match = re.compile('([a-zA-Z\.\/]+)').search(ticket)
    if match:
        return match.group()
    else:
        return 'U'

def getTicketNumber(ticket):
    match = re.compile('([\d]+$)').search(ticket)
    if match:
        return match.group()
    else:
        return '0'

def processFare():
    global df
    df['Fare'][ np.isnan(df['Fare']) ] = df['Fare'].median() # 缺失值取中位数 # 只有在测试集中有一个缺失值
    df['Fare'][ np.where(df['Fare'] == 0)[0] ] = df['Fare'][ df['Fare'].nonzero()[0] ].min() / 10# 为什么用最小值补充呢？
	# 不该用最小值补充票价，而是应该分层级补充票价。
    # 虽然应该分层级补充票价，但是通过观察发现，这些票价为0的人存活率很低，所以直接用原程序的方法应该也行。
    df['Fare_bin'] = pd.qcut(df['Fare'], 4)
    if keep_binary:
        df = pd.concat([ df, pd.get_dummies(df['Fare_bin']).rename(columns = lambda x: 'Fare_' + str(x)) ], axis = 1)

    if keep_bins:
        df['Fare_bin_id'] = pd.factorize(df['Fare_bin'])[0] + 1  # 保留编号

    if keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['Fare_scaled'] = scaler.fit_transform(df['Fare'])

    if keep_bins and keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['Fare_bin_id_scaled'] = scaler.fit_transform(df['Fare_bin_id'])  # 把编号也标准化


    if not keep_strings:
        df.drop('Fare_bin', axis = 1, inplace = True)

def processEmbarked():
    global df
    df.Embarked[ df.Embarked.isnull() ] = df.Embarked.dropna().mode().values  #　mode函数返回个数最多的项（可多个）构成的series，
    df['Embarked'] = pd.factorize(df['Embarked'])[0]
    if keep_binary:
        df = pd.concat([ df, pd.get_dummies(df['Embarked']).rename(columns = lambda x: 'Embarked_' + str(x)) ], axis = 1)

def processPClass():
    global df
    df.Pclass[ df.Pclass.isnull() ] = df.Pclass.dropna().mode().values

    if keep_binary:
        df = pd.concat([ df, pd.get_dummies(df['Pclass']).rename(columns = lambda x: 'Pclass_' + str(x)) ], axis = 1)

    if keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['Pclass_scaled'] = scaler.fit_transform(df['Pclass'])

def processFamily():
    global df
    df['SibSp'] = df['SibSp'] + 1
    df['Parch'] = df['Parch'] + 1
    if keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['SibSp_scaled'] = scaler.fit_transform(df['SibSp'])
        df['Parch_scaled'] = scaler.fit_transform(df['Parch'])
    if keep_binary:
        sibsps = pd.get_dummies(df['SibSp']).rename(columns = lambda x: 'SibSp_' + str(x))
        parchs = pd.get_dummies(df['Parch']).rename(columns = lambda x: 'Parch_' + str(x))
        df = pd.concat([df, sibsps, parchs], axis = 1)

def processSex():
    global df
    df['Gender'] = np.where(df['Sex'] == 'male', 1, 0)
	# 用get_dummies会不会好一些？

def processName():
    global df

    df['Names'] = df['Name'].map(lambda x: len(re.split(' ', x)))


    df['Title'] = df['Name'].map(lambda x: re.compile(', (.*?)\.').findall(x)[0])


    df['Title'][df.Title == 'Jonkheer'] = 'Master'
    df['Title'][df.Title.isin(['Ms', 'Mlle'])] = 'Miss'
    df['Title'][df.Title == 'Mme'] = 'Mrs'
    df['Title'][df.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir'])] = 'Sir'
    df['Title'][df.Title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'
    # 可以看出至少可以拆分为六种特征：姓名长度，Title称谓，get_dummies后的title，
    # 标准化后的姓名长度，factorize后的title类别，标准化的factorize后的title类别。

    if keep_binary:
        df = pd.concat([ df, pd.get_dummies(df['Title']).rename(columns = lambda x: 'Title_' + str(x)) ], axis = 1)


    if keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['Names_scaled'] = scaler.fit_transform(df['Names'])

    if keep_bins:
        df['Title_id'] = pd.factorize(df['Title'])[0] + 1

    if keep_bins and keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['Title_id_scaled'] = scaler.fit_transform(df['Title_id'])

def processAge():
    global df
    setMissingAges()

    if keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['Age_scaled'] = scaler.fit_transform(df['Age'])

    df['isChild'] = np.where(df.Age < 13, 1, 0) # 小于13的用1表示，否则用0表示

    df['Age_bin'] = pd.qcut(df['Age'], 4)
    if keep_binary:
        df = pd.concat([ df, pd.get_dummies(df['Age_bin']).rename(columns = lambda x: 'Age_' + str(x)) ], axis = 1)

    if keep_bins:
        df['Age_bin_id'] = pd.factorize(df['Age_bin'])[0] + 1

    if keep_bins and keep_scaled:
        scaler = preprocessing.StandardScaler()
        df['Age_bin_id_scaled'] = scaler.fit_transform(df['Age_bin_id'])

    if not keep_strings:
        df.drop('Age_bin', axis = 1, inplace = True)

def setMissingAges():
    global df

    age_df = df[[ 'Age', 'Embarked', 'Fare', 'Parch', 'SibSp', 'Title_id', 'Pclass', 'Names', 'CabinLetter' ]]
    X = age_df.loc[ (df.Age.notnull()) ].values[:, 1::]
    y = age_df.loc[ (df.Age.notnull()) ].values[:, 0]

    rtr = RandomForestRegressor(n_estimators = 2000, n_jobs = -1)
    rtr.fit(X, y)

    predictedAges = rtr.predict(age_df.loc[ (df.Age.isnull()) ].values[:, 1::])
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges  # 应该转化为整数

def processDrops():
    global df

    rawDropList = [ 'Name', 'Names', 'Title', 'Sex', 'SibSp', 'Parch', 'Pclass', 'Embarked', \
                    'Cabin', 'CabinLetter', 'CabinNumber', 'Age', 'Fare', 'Ticket', 'TicketNumber' ]
    stringsDropList = [ 'Title', 'Name', 'Cabin', 'Ticket', 'Sex', 'Ticket', 'TicketNumber' ]
    if not keep_raw:
        df.drop(rawDropList, axis = 1, inplace = True)
    elif not keep_strings:
        df.drop(stringsDropList, axis = 1, inplace = True)

def getDataSets(binary = False, bins = False, scaled = False, strings = False, raw = True, pca = False, balanced = False):
    global keep_binary, keep_bins, keep_scaled, keep_raw, keep_strings, df

    keep_binary = binary
    keep_bins  = bins
    keep_scaled = scaled
    keep_raw = raw
    keep_strings = strings

    input_df = pd.read_csv('train.csv', header = 0)
    submit_df = pd.read_csv('test.csv', header = 0)
    df = pd.concat([input_df, submit_df]) # 非共有属性变为nan，合并的目的是对两个数据集进行相同的操作
    df.reset_index(inplace = True)
    df.drop('index', axis = 1, inplace = True)
    df = df.reindex_axis(input_df.columns, axis = 1) # 按照input_df的属性排列方式排列属性
    processCabin()
    processTicket()
    processName()
    processFare()
    processEmbarked()
    processFamily()
    processSex()
    processPClass()
    processAge()
    processDrops()
    columns_list = list(df.columns.values)
    columns_list.remove('Survived')
    new_col_list = list(['Survived'])
    new_col_list.extend(columns_list)
    df = df.reindex(columns = new_col_list)  # 重新排列属性

    print 'Starting with', df.columns.size, 'manually generated features...\n', df.columns.values
    numerics = df.loc[:, [ 'Age_scaled', 'Fare_scaled', 'Pclass_scaled', 'Parch_scaled', 'SibSp_scaled', \
                            'Names_scaled', 'CabinNumber_scaled', 'Age_bin_id_scaled', 'Fare_bin_id_scaled' ]]
	#　只对上面list中的属性进行组合，其他的因为进行了get_dummies，所以组合不了。
    print '\nFeatures used for automated feature generation:\n', numerics.head(10)

    new_fields_count = 0
    for i in range(0, numerics.columns.size-1):
        for j in range(0, numerics.columns.size-1):
            if i <= j:  #　因为自己乘以自己有意义，意味着权重更大，并且乘法有交换性，所以得用i<=j
                name = str(numerics.columns.values[i]) + '*' + str(numerics.columns.values[j])
                df = pd.concat([ df, pd.Series(numerics.iloc[:, i] * numerics.iloc[:, j], name = name) ], axis = 1)
                new_fields_count += 1
            if i < j:  # 自己加自己没有意义，并且加法有交换性，所以得用i<j
                name = str(numerics.columns.values[i]) + '+' + str(numerics.columns.values[j])
                df = pd.concat([ df, pd.Series(numerics.iloc[:, i] + numerics.iloc[:, j], name = name) ], axis = 1)
                new_fields_count += 1
            if not i == j: # 自己减（除以）自己没有意义，减法和除法没有交换性，所以用 i != j
                name = str(numerics.columns.values[i]) + '/' + str(numerics.columns.values[j])
                df = pd.concat([ df, pd.Series(numerics.iloc[:, i] / numerics.iloc[:, j], name = name) ], axis = 1)
                name = str(numerics.columns.values[i]) + '-' + str(numerics.columns.values[j])
                df = pd.concat([ df, pd.Series(numerics.iloc[:, i] - numerics.iloc[:, j], name = name) ], axis = 1)
                new_fields_count += 2

    print '\n', new_fields_count, 'new features generated'
    df_corr = df.drop(['Survived', 'PassengerId'], axis = 1).corr(method = 'spearman')

    # 这两步是为了去除属性与其本身的相关系数
    mask = np.ones(df_corr.columns.size) - np.eye(df_corr.columns.size)
    df_corr = mask * df_corr

    # 对于每个属性，找到与其相关系数大于0.98的属性，并且是对称矩阵，所以只找一次就行。
    drops = []
    for col in df_corr.columns.values:
        if np.in1d([col], drops):
            continue
        corr = df_corr[abs(df_corr[col]) > 0.98].index
        drops = np.union1d(drops, corr)

    # 去掉相关性很高的属性
    print '\nDropping', drops.shape[0], 'highly correlated features...\n' #, drops
    df.drop(drops, axis = 1, inplace = True)

    input_df = df[ :input_df.shape[0] ]
    submit_df = df[ input_df.shape[0]: ]

    # 进行PCA降维和聚类，不过聚类是什么作用呢？
    if pca:
        print 'reducing and clustering now...'
        input_df, submit_df = reduceAndCluster(input_df, submit_df)
    else:
        submit_df.drop('Survived', axis = 1, inplace = 1)

    print '\n', input_df.columns.size, 'initial features generated...\n' #, input_df.columns.values

    # 因为负样例太多，所以需要对训练样本的负样本进行负采样
    if balanced:
        print 'Perished data shape:', input_df[input_df.Survived == 0].shape
        print 'Survived data shape:', input_df[input_df.Survived == 1].shape
        perished_sample = rd.sample( input_df[input_df.Survived == 0].index, input_df[input_df.Survived == 1].shape[0] )
        input_df = pd.concat( [input_df.ix[perished_sample], input_df[input_df.Survived == 1]] )
        input_df.sort(inplace = True)
        print 'New even class training shape:', input_df.shape

    return input_df, submit_df

def reduceAndCluster(input_df, submit_df, clusters = 3):

    df = pd.concat([input_df, submit_df])
    df.reset_index(inplace = True)
    df.drop('index', axis = 1, inplace = True)
    df = df.reindex_axis(input_df.columns, axis = 1)
    survivedSeries = pd.Series(df['Survived'], name = 'Survived')

    print df.head()
    X = df.values[:, 1::]
    y = df.values[:, 0]

    print X[0:5]

    variance_pct = .99

    pca = PCA(n_components = variance_pct)

    X_transformed = pca.fit_transform(X, y)

    pcaDataFrame = pd.DataFrame(X_transformed) #　降维后的ＤataFrame

    print pcaDataFrame.shape[1], ' components describe ', str(variance_pct)[1:], '% of the variance'


    # 疑问：计算聚类中心有什么用？
    kmeans = KMeans(n_clusters = clusters, random_state = np.random.RandomState(4), init = 'random')
    trainClusterIds = kmeans.fit_predict(X_transformed[:input_df.shape[0]])
    print 'clusterIds shape for training data: ', trainClusterIds.shape

    testClusterIds = kmeans.predict(X_transformed[ input_df.shape[0]: ])
    print 'clusterIds shape for test data: ', testClusterIds.shape

    clusterIds = np.concatenate([trainClusterIds, testClusterIds])
    print 'all clusterIds shape: ', clusterIds.shape

    clusterIdSeries = pd.Series(clusterIds, name = 'ClusterId')
    df = pd.concat([survivedSeries, clusterIdSeries, pcaDataFrame], axis = 1)

    input_df = df[ :input_df.shape[0] ]
    submit_df = df[ input_df.shape[0]: ]
    submit_df.reset_index(inplace = True)
    submit_df.drop('index', axis = 1, inplace = True)
    submit_df.drop('Survived', axis = 1, inplace = 1)

    return input_df, submit_df

# 这个是什么意思？
if __name__ == '__main__':
    train, test = getDataSets(bins = True, scaled = True, binary = True)
    drop_list = ['PassengerId']
    train.drop(drop_list, axis = 1, inplace = 1)
    test.drop(drop_list, axis = 1, inplace = 1)

    train, test = reduceAndCluster(train, test) #　为什么再次进行ＰＣＡ和聚类？

    print 'Labeled survived counts :\n', pd.value_counts(train['Survived']) / train.shape[0]
    print 'Labeles cluster counts  :\n', pd.value_counts(train['ClusterId']) / train.shape[0]
    print 'Unlabeled cluster counts:\n', pd.value_counts(test['ClusterId']) / test.shape[0]

    print train.columns.values






















