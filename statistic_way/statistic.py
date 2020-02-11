#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12) 
 
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

def create_data():
    '''
     创建训练数据和测试数据
     :return: X_train:训练集， X_test:测试集
    '''
    np.random.seed(42)  # 设置seed使每次生成的随机数都相等
    m, s = 3, 0.1 # 设置均值和方差
    X_train = np.random.normal(m, s, 100) # 100个一元正态分布数据
    # 构造10测试数据，从一个均匀分布[low,high)中随机采样
    X_test = np.random.uniform(low=m - 1, high=m + 1, size=10)
    return X_train, X_test
 
def plot_data(X_train, X_test):
    '''
    数据可视化
    :param X_train: 训练集
    :param X_test: 测试集
    :return:
    '''
    fig = plt.figure(figsize=(10, 4))
    plt.subplots_adjust(wspace=0.5)  # 调整子图之间的左右边距
    fig.add_subplot(1, 2, 1)  # 绘制训练数据的分布
    plt.scatter(X_train, [0] * len(X_train), color='blue', marker='x', label='train data')
    plt.title('train data plot')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')

    fig.add_subplot(1, 2, 2)  # 绘制整体数据的分布
    plt.scatter(X_train, [0] * len(X_train), color='blue', marker='x', label='train data')
    plt.scatter(X_test, [0] * len(X_test), color='red', marker='^',label='test data')
    plt.title('all data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')

    plt.show()

def fit(X_train):
    '''
     拟合数据，训练模型
     :param X_train: 训练集
     :return:  mu:均值, sigma:方差
    '''
    global mu, sigma
    mu = np.mean(X_train)  # 计算均值μ
    sigma = np.var(X_train) # 计算方差 σ^2
 
def gaussian(X):
    '''
     计算正态分布
     :param X: 数据集
     :return: 数据集的密度值
    '''
    return np.exp(-((X - mu) ** 2) / (2 * sigma)) / (np.sqrt(2 * np.pi) * np.sqrt(sigma))
   
def get_epsilon(n=3):
    ''' 调整ε的值，默认ε=3σ '''
    return np.sqrt(sigma) * n

def predict(X):
    '''
     检测训练集中的数据是否是正常数据
     :param X: 待预测的数据
     :return: P1:数据的密度值, P2:数据的异常检测结果，True正常，False异常
    '''
    P1 = gaussian(X) # 数据的密度值
    epsilon = get_epsilon()
    P2 = [p > epsilon for p in P1] # 数据的异常检测结果，True正常，False异常
    return P1, P2

def plot_predict(X):
    '''可视化异常检测结果 '''
    epsilon = get_epsilon()
    xs = np.linspace(mu - epsilon, mu + epsilon, 50)
    ys = gaussian(xs)
    plt.plot(xs, ys, c='g', label='fitting curve')  # 绘制正态分布曲线

    P1, P2 = predict(X)
    normals_idx = [i for i, t in enumerate(P2) if t == True] # 正常数据的索引
    plt.scatter([X[i] for i in normals_idx], [P1[i] for i in normals_idx],
                             color='blue', marker='x', label='noraml data')
    outliers_idx = [i for i, t in enumerate(P2) if t == False] # 异常数据的索引
    plt.scatter([X[i] for i in outliers_idx], [P1[i] for i in outliers_idx],
                             color='red', marker='^', label='abnoraml data')
    plt.title('detect res {} abnoraml data'.format(len(outliers_idx)))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper left')
    plt.show()
    
if __name__ == '__main__':
    mu, sigma = 0, 0 # 模型的均值μ和方差σ^2
    X_train, X_test = create_data()
    plot_data(X_train, X_test)
    fit(X_train)
    print('μ = {}, σ^2 = {}'.format(mu, sigma))
    plot_predict(np.r_[X_train, X_test])
