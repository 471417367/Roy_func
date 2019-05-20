# -*- coding: utf-8 -*-
"""
@author: Roy
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号

# 返回重复的数据行
# df[df['col'].duplicated(keep=False)]
# 删除其中一个重复user_id行
# df = df.drop_duplicates(subset=['col'])