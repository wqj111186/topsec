import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_anomalies import *
from sklearn.datasets import load_breast_cancer

# 载入数据
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
print(type(cancer))
print(type(X))
print(type(y))
print('data shape: {0};target shape: {1} no. positive: {2}; no. negative: {3}'.format(
    X.shape, y.shape,y[y==1].shape[0], y[y==0].shape[0])) #shape[0]就是读取矩阵第一维度的长度
print(cancer.data[0])  #打印一组样本数据

#df.head(2)
print(len(cancer.feature_names))
print(cancer.feature_names)

#value_counts()
print(np.unique(y))

plt.rcParams['figure.figsize'] = [12, 8]
plot_anomalies(X, y, sample_size=5, n_trees=1000, 
               desired_TPR=0.75, improved=True)

def sklearn_to_df(sklearn_dataset):
    df = pd.DataFrame(sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    df['target'] = pd.Series(sklearn_dataset.target)
    return df

df = sklearn_to_df(cancer)
add_noise(df, 5)
print(df.head(2))

X, y = df.drop('target', axis=1), df['target']
plot_anomalies(X, y, sample_size=5, n_trees=1000, 
               desired_TPR=0.75, improved=True)