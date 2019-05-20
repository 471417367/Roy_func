# -*- coding: utf-8 -*-
"""
@author: Roy
"""

from scipy.interpolate import lagrange

# 自定义插值函数
def ployinterp_column(s, n, k=5):
    y = s[list(range(n-k, n)) + list(range(n+1, n+1+k))]
    y = y[y.notnull()]
    return lagrange(y.index, list(y))(n)

# 逐个元素判断是否需要插值
# 所有列的NaN进行lagrange进行差值
def lagrange_df(df):
    for i in df.columns:
        for j in range(len(df)):
            if (df[i].isnull())[j]:
                df[i][j] = ployinterp_column(df[i], j)
                
# 指定列的NaN进行lagrange进行插值
def lagrange_df_col(df_col):
    for j in range(len(df_col)):
        if (df_col.isnull())[j]:
            df_col[j] = ployinterp_column(df_col, j)             