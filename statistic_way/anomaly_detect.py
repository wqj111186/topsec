
import pandas as pd
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import PC
# 载入数据
def load_data(file):
    '''
     process kdd data for detection 
    '''
    df = pd.read_csv(file, header=None)
    print(df.describe())
    print('df shape: ', df.shape)
    print('df shape: ', df.columns)
    
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(df.iloc[:, -1].values.tolist())
    le = preprocessing.LabelEncoder()
    df.iloc[:, 1] = le.fit_transform(df.iloc[:, 1].values.tolist())
    df.iloc[:, 2] = le.fit_transform(df.iloc[:, 2].values.tolist())
    df.iloc[:, 3] = le.fit_transform(df.iloc[:, 3].values.tolist())
    X = df.iloc[:, :-1]
        
    return X, y

X, y = load_data('kddcup.csv')
print(type(X))
print(type(y))
print(X.isnull().any())
print(X[X.isnull().values==True])
print(X.columns[X.isnull().any()].tolist())
print(X.isnull().sum())
#print('data shape: {0};target shape: {1} no. positive: {2}; no. negative: {3}'.format(
    #X.shape, y.shape,y[y==1].shape[0], y[y==0].shape[0])) #shape[0]就是读取矩阵第一维度的长度
print(X.iloc[0,:])  #打印一组样本数据

#value_counts()
print(np.unique(y))

def plot_data():
    plt.figure(figsize=(8,5))
    plt.plot(X[:,0], X[:,1], 'bx')
    # plt.scatter(Xval[:,0], Xval[:,1], c=yval.flatten(), marker='x', cmap='rainbow')
    
plot_data()

def gaussian(X, mu, sigma2):
    '''
    mu, sigma2参数已经决定了一个高斯分布模型
    因为原始模型就是多元高斯模型在sigma2上是对角矩阵而已，所以如下：
    If Sigma2 is a matrix, it is treated as the covariance matrix. 
    If Sigma2 is a vector, it is treated as the sigma^2 values of the variances
    in each dimension (a diagonal covariance matrix)
    output:
        一个(m, )维向量，包含每个样本的概率值。
    '''

    m, n = X.shape
    if np.ndim(sigma2) == 1:
        sigma2 = np.diag(sigma2)

    norm = 1./(np.power((2*np.pi), n/2)*np.sqrt(np.linalg.det(sigma2)))
    exp = np.zeros((m,1))
    for row in range(m):
        xrow = X[row]
        exp[row] = np.exp(-0.5*((xrow-mu).T).dot(np.linalg.inv(sigma2)).dot(xrow-mu))
    return norm*exp


def getGaussianParams(X, useMultivariate):
    """
    The input X is the dataset with each n-dimensional data point in one row
    The output is an n-dimensional vector mu, the mean of the data set 
    the variances sigma^2, an n x 1 vector 或者是(n,n)矩阵，if你使用了多元高斯函数
    求样本方差除的是 m 而不是 m - 1，实际上效果差不了多少。
    """
    mu = X.mean(axis=0)
    if useMultivariate:    
        sigma2 = ((X-mu).T @ (X-mu)) / len(X)
    else:
        sigma2 = X.var(axis=0, ddof=0)  # 样本方差
    
    return mu, sigma2


def plotContours(mu, sigma2):
    """
    画出高斯概率分布的图，在三维中是一个上凸的曲面。投影到平面上则是一圈圈的等高线。
    """
    delta = .3  # 注意delta不能太小！！！否则会生成太多的数据，导致矩阵相乘会出现内存错误。
    x = np.arange(0,30,delta)
    y = np.arange(0,30,delta)
    
    # 这部分要转化为X形式的坐标矩阵，也就是一列是横坐标，一列是纵坐标，
    # 然后才能传入gaussian中求解得到每个点的概率值
    xx, yy = np.meshgrid(x, y)
    points = np.c_[xx.ravel(), yy.ravel()]  # 按列合并，一列横坐标，一列纵坐标
    z = gaussian(points, mu, sigma2)
    z = z.reshape(xx.shape)  # 这步骤不能忘
    
    cont_levels = [10**h for h in range(-20,0,3)]
    plt.contour(xx, yy, z, cont_levels)  # 这个levels是作业里面给的参考,或者通过求解的概率推出来。

    plt.title('Gaussian Contours',fontsize=16)    
        
    # First contours without using multivariate gaussian:
    plot_data()
    useMV = False
    plotContours(*getGaussianParams(X, useMV))
    
    # Then contours with multivariate gaussian:
    plot_data()
    useMV = True
    # *表示解元组
    plotContours(*getGaussianParams(X, useMV))

def selectThreshold(yval, pval):
    def computeF1(yval, pval):
        m = len(yval)
        tp = float(len([i for i in range(m) if pval[i] and yval[i]]))
        fp = float(len([i for i in range(m) if pval[i] and not yval[i]]))
        fn = float(len([i for i in range(m) if not pval[i] and yval[i]]))
        prec = tp/(tp+fp) if (tp+fp) else 0
        rec = tp/(tp+fn) if (tp+fn) else 0
        F1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0
        return F1

    epsilons = np.linspace(min(pval), max(pval), 1000)
    bestF1, bestEpsilon = 0, 0
    for e in epsilons:
        pval_ = pval < e
        thisF1 = computeF1(yval, pval_)
        if thisF1 > bestF1:
            bestF1 = thisF1
            bestEpsilon = e

    return bestF1, bestEpsilon
###low dimensional dataset
#mu, sigma2 = getGaussianParams(X, useMultivariate=False)
#pval = gaussian(Xval, mu, sigma2)
#bestF1, bestEpsilon = selectThreshold(yval, pval)

#y = gaussian(X, mu, sigma2)  # X的概率
#xx = np.array([X[i] for i in range(len(y)) if y[i] < bestEpsilon])

#plot_data()
#plotContours(mu, sigma2)
#plt.scatter(xx[:,0], xx[:,1], s=80, facecolors='none', edgecolors='r')

####High dimensional dataset
mu, sigma2 = getGaussianParams(X2, useMultivariate=False)
ypred = gaussian(X2, mu, sigma2)

yval2pred = gaussian(Xval2, mu, sigma2)
# You should see a value epsilon of about 1.38e-18, and 117 anomalies found.
bestF1, bestEps = selectThreshold(yval2, yval2pred)
anoms = [X2[i] for i in range(X2.shape[0]) if ypred[i] < bestEps]
print(bestEps, len(anoms))