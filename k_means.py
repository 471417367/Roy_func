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

# K-means聚类模型
from sklearn.cluster import KMeans
# 假设我需要聚类出k个不同类别的客户，创建分类器，并训练它。
# feature = ['飞行总次数F', '总飞行公里数M', '平均折扣率C', '第二年飞行次数与第一年飞行次数的比例R']
# feature = [df.columns[0], df.columns[1], df.columns[2], df.columns[3]]
def k_means(df, k, feature):
    km = KMeans(n_clusters =k, n_jobs = 4).fit(df)
    
    r1 = pd.Series(km.labels_).value_counts() #统计各个类别的数目
    r2 = pd.DataFrame(km.cluster_centers_) #找出聚类中心
    # 所有簇中心坐标值中最大值和最小值
    max = r2.values.max()
    min = r2.values.min()
    r = pd.concat([r2, r1], axis = 1) #横向连接（0是纵向），得到聚类中心对应的类别下的数目
    r.columns = list(df.columns) + [u'类别数目'] #重命名表头
     
    # 绘图这里有一个网址可以参考他的作图https://blog.csdn.net/a857553315/article/details/79177524
    fig=plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    center_num = r.values
    
    N =len(feature)
    for i, v in enumerate(center_num):
        # 设置雷达图的角度，用于平分切开一个圆面
        angles=np.linspace(0, 2*np.pi, N, endpoint=False)
        # 为了使雷达图一圈封闭起来，需要下面的步骤
        center = np.concatenate((v[:-1],[v[0]]))
        angles = np.concatenate((angles,[angles[0]]))
        # 绘制折线图
        ax.plot(angles, center, 'o-', linewidth=2, label = "第%d簇人群,%d人"% (i+1,v[-1]))
        # 填充颜色
        ax.fill(angles, center, alpha=0.25)
        # 添加每个特征的标签
        ax.set_thetagrids(angles * 180/np.pi, feature, fontsize=15)
        # 设置雷达图的范围
        ax.set_ylim(min-0.1, max+0.1)
        # 添加标题
        plt.title('客户群特征分析图', fontsize=20)
        # 添加网格线
        ax.grid(True)
        # 设置图例
        plt.legend(loc='upper right', bbox_to_anchor=(1.3,1.0),ncol=1,fancybox=True,shadow=True)
        
    # 显示图形
    plt.show()
    