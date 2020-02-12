import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.datasets import make_moons, make_blobs
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

print(__doc__)

matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

# Example settings
n_samples = 300
outliers_fraction = 0.15
n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers

# define outlier/anomaly detection methods to be compared
# 四种异常检测算法
anomaly_algorithms = [
    ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
    ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                                      gamma=0.1)),
    ("Isolation Forest", IsolationForest(contamination=outliers_fraction,
                                         random_state=42)),
    ("Local Outlier Factor", LocalOutlierFactor(
        n_neighbors=35, contamination=outliers_fraction))]

# Define datasets
blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)
datasets = [
    # make_blobes用于生成聚类数据。centers表示聚类中心，cluster_std表示聚类数据方差。返回值(数据, 类别)
    # **用于传递dict key-value参数，*用于传递元组不定数量参数。
    make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5,
               **blobs_params)[0],
    make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[0.5, 0.5],
               **blobs_params)[0],
    make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[1.5, .3],
               **blobs_params)[0],

    # make_moons用于生成月亮形数据。返回值数据(x, y)
    4. * (make_moons(n_samples=n_samples, noise=.05, random_state=0)[0] -
          np.array([0.5, 0.25])),
    14. * (np.random.RandomState(42).rand(n_samples, 2) - 0.5)]

# Compare given classifiers under given settings
# np.meshgrid生产成网格数据
# 如输入x = [0, 1, 2, 3] y = [0, 1, 2]，则输出
# xx 0 1 2 3   yy 0 0 0 0
#    0 1 2 3      1 1 1 1
#    0 1 2 3      2 2 2 2
xx, yy = np.meshgrid(np.linspace(-7, 7, 150),
                     np.linspace(-7, 7, 150))

# figure生成画布，subplots_adjust子图的间距调整，左边距，右边距，下边距，上边距，列间距，行间距
plt.figure(figsize=(len(anomaly_algorithms) * 2 + 3, 12.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

plot_num = 1
rng = np.random.RandomState(42)

for i_dataset, X in enumerate(datasets):
    # Add outliers
    # np.concatenate数组拼接。axis=0行增加，axis=1列增加（对应行拼接）。
    X = np.concatenate([X, rng.uniform(low=-6, high=6,
                                       size=(n_outliers, 2))], axis=0)

    for name, algorithm in anomaly_algorithms:
        t0 = time.time()

        algorithm.fit(X)
        t1 = time.time()
        # 定位子图位置。参数：列，行，序号
        plt.subplot(len(datasets), len(anomaly_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)

        # fit the data and tag outliers
        # 训练与预测
        if name == "Local Outlier Factor":
            y_pred = algorithm.fit_predict(X)
        else:
            y_pred = algorithm.fit(X).predict(X)

        # plot the levels lines and the points
        # 用训练的模型预测网格数据点，主要是要得到聚类模型边缘
        if name != "Local Outlier Factor":  # LOF does not implement predict
            # ravel()多维数组平铺为一维数组。np.c_ cloumn列连接，np.r_ row行连接。
            Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
            # reshape这里把一维数组转化为二维数组
            Z = Z.reshape(xx.shape)
            # plt.contour画等高线。Z表示对应点类别，可以理解为不同的高度，plt.contour就是要画出不同高度间的分界线。
            plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

        colors = np.array(['#377eb8', '#ff7f00'])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(y_pred + 1) // 2])

        # x轴范围
        plt.xlim(-7, 7)
        plt.ylim(-7, 7)
        # x轴坐标
        plt.xticks(())
        plt.yticks(())
        # 坐标图上显示的文字
        #plt.text(.99, .01, ('),
                            #transform=plt.gca().transAxes, size=15,
                 #horizontalalignment='right')
                 #plot_num += 1

plt.show()