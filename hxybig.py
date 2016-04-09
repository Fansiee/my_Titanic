# -*- coding: utf-8 -*-
"""
Created on Thu Apr 07 14:01:42 2016

@author: fanqi
"""
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.datasets import load_digits
import numpy as np
from sklearn import (manifold, decomposition, random_projection)
from matplotlib import offsetbox
import time

digits = load_digits(n_class = 6)
X = digits.data
y = digits.target
n_samples, n_features = X.shape
print 'Dataset consist of %d samples with %d features each' % (n_samples, n_features)

n_img_per_row = 20
img = np.zeros((10 * n_img_per_row, 10 * n_img_per_row))
for i in range(n_img_per_row):
    ix = 10 * i + 1
    for j in range(n_img_per_row):
        iy = 10 * j + 1
        img[ix: ix + 8, iy: iy + 8] = X[i * n_img_per_row + j].reshape((8, 8))

plt.imshow(img, cmap=plt.cm.binary)
plt.xticks([])
plt.yticks([])
_ = plt.title('A selection from the 8*8=64-dimensional digits dataset')
plt.show()

rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)

def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(digits.target[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 12})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

#记录开始时间
start_time = time.time()
X_projected = rp.fit_transform(X)
plot_embedding(X_projected, "Random Projection of the digits (time: %.3fs)" % (time.time() - start_time))

#TruncatedSVD 是 PCA的一种实现
X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
#记录时间
start_time = time.time()
plot_embedding(X_pca,"Principal Components projection of the digits (time: %.3fs)" % (time.time() - start_time))


#降维
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
start_time = time.time()
X_tsne = tsne.fit_transform(X)
#绘图
plot_embedding(X_tsne,
               "t-SNE embedding of the digits (time: %.3fs)" % (time.time() - start_time))


import numpy as np
import matplotlib.plot as plt
# 改自http://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_loss_functions.html
xmin, xmax = -4, 4
xx = np.linspace(xmin, xmax, 100)
plt.plot([xmin, 0, 0, xmax], [1, 1, 0, 0], 'k-',
         label="Zero-one loss")
plt.plot(xx, np.where(xx < 1, 1 - xx, 0), 'g-',
         label="Hinge loss")
plt.plot(xx, np.log2(1 + np.exp(-xx)), 'r-',
         label="Log loss")
plt.plot(xx, np.exp(-xx), 'c-',
         label="Exponential loss")
plt.plot(xx, -np.minimum(xx, 0), 'm-',
         label="Perceptron loss")

plt.ylim((0, 8))
plt.legend(loc="upper right")
plt.xlabel(r"Decision function $f(x)$")
plt.ylabel("$L(y, f(x))$")
plt.show()























'''
X, y = make_classification(200000, n_features = 200, n_informative = 25, n_redundant = 0, n_classes = 10, class_sep = 2, random_state = 0)

est = SGDClassifier(penalty = 'l2', alpha = 0.001)
progressive_validation_score = []
train_score = []
for datapoint in range(0, 199000, 1000):
    X_batch = X[datapoint: datapoint + 1000]
    y_batch = y[datapoint: datapoint + 1000]
    if datapoint > 0:
        progressive_validation_score.append(est.score(X_batch, y_batch))
    est.partial_fit(X_batch, y_batch, classes = range(10))
    if datapoint > 0:
        train_score.append(est.score(X_batch, y_batch))

plt.plot(train_score, label = 'train score')
plt.plot(progressive_validation_score, label = 'progressive validation score')
plt.xlabel('Mini-batch')
plt.ylabel('Score')
plt.legend(loc = 'best')
plt.show()
'''


