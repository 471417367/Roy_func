# -*- coding: utf-8 -*-
"""
@author: Roy
"""

# 数据标准化
from sklearn.preprocessing import StandardScaler
# 数据归一化
from sklearn.preprocessing import MinMaxScaler

# 标准化数据，保证每个维度的特征数据方差为1，均值为0，
# 使得预测结果不会被某些维度过大的特征值而主导
def standard(df):
    ss = StandardScaler()
    # fit_transform()先拟合数据，再标准化
    df_std = ss.fit_transform(df)
    # transform()数据标准化
    # X_test = ss.transform(X_test)
    
    return df_std

# 归一化是利用特征的最大最小值，将特征的值缩放到[0,1]区间
# 对于每一列的特征使用min-max函数进行缩放
def minmax(df):   
    minMax = MinMaxScaler()
    #将数据进行归一化
    df_mm = minMax.fit_transform(df)
    
    return df_mm