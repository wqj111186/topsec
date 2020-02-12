#-*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from scipy import io as spio
from sklearn.decomposition import pca

def PCA_2D(X):
    #data_2d = spio.loadmat("data.mat")
    #X = data_2d['X']
    m = X.shape[0]
    plt = plot_data_2d(X, 'bo') # 显示二维的数据
    plt.show()
    
    X_copy = X.copy()
    X_norm, mu, sigma = featureNormalize(X_copy)    # 归一化数据
    #plot_data_2d(X_norm)    # 显示归一化后的数据
    #plt.show()
    
    Sigma = np.dot(np.transpose(X_norm), X_norm)/m  # 求Sigma
    U, S, V = np.linalg.svd(Sigma)       # 求Sigma的奇异值分解
    
    plt = plot_data_2d(X, 'bo') # 显示原本数据
    drawline(plt, mu, mu+S[0]*(U[:,0]), 'r-')  # 线，为投影的方向

    plt.axis('square')
    plt.show()
    
    K = 1  # 定义降维多少维（本来是2维的，这里降维1维）
    '''投影之后数据（降维之后）'''
    Z = projectData(X_norm,U,K)   # 投影
    '''恢复数据'''
    X_rec = recoverData(Z,U,K)    # 恢复
    '''作图-----原数据与恢复的数据'''
    plt = plot_data_2d(X_norm,'bo')
    plot_data_2d(X_rec,'ro')
    for i in range(X_norm.shape[0]):
        drawline(plt, X_norm[i,:], X_rec[i,:], '--k')
    plt.axis('square')
    plt.show()
       
# 可视化二维数据
def plot_data_2d(X,marker):
    plt.plot(X[:,0],X[:,1],marker) 
    return plt

# 归一化数据
def featureNormalize(X):
    '''（每一个数据-当前列的均值）/当前列的标准差'''
    n = X.shape[1]
    mu = np.zeros((1,n));
    sigma = np.zeros((1,n))
    
    mu = np.mean(X,axis=0)   # axis=0表示列
    sigma = np.std(X,axis=0)
    for i in range(n):
        X[:,i] = (X[:,i]-mu[i])/sigma[i]
    return X,mu,sigma


# 映射数据
def projectData(X_norm,U,K):
    Z = np.zeros((X_norm.shape[0],K))
    
    U_reduce = U[:,0:K]          # 取前K个
    Z = np.dot(X_norm,U_reduce) 
    return Z

# 画一条线
def drawline(plt,p1,p2,line_type):
    plt.plot(np.array([p1[0],p2[0]]),np.array([p1[1],p2[1]]),line_type)  

# 恢复数据
def recoverData(Z, U, K):
    X_rec = np.zeros((Z.shape[0],U.shape[0]))
    U_recude = U[:, 0 : K]
    X_rec = np.dot(Z, np.transpose(U_recude))  # 还原数据（近似）
    return X_rec

if __name__ == "__main__":
    PCA_2D()