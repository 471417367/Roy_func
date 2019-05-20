# -*- coding: utf-8 -*-
"""
@author: Roy
"""
import numpy as np
from sklearn.model_selection import train_test_split

def train_test(df, size):
    df_col = np.array(df.columns)
    x = df[df_col[1:]]
    y = df[df_col[0]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                        test_size = size, 
                                                        random_state=0)
    
    return x_train, x_test, y_train, y_test