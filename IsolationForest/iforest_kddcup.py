import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_anomalies import *
from sklearn import preprocessing

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

plt.rcParams['figure.figsize'] = [12, 8]
plot_anomalies(X, y, sample_size=5, n_trees=1000, 
               desired_TPR=0.75, improved=True)

#add_noise(X, 5)
#print(X.head(2))

#plot_anomalies(X, y, sample_size=5, n_trees=1000, 
               #desired_TPR=0.75, improved=True)