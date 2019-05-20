# -*- coding: utf-8 -*-
"""
@author: Roy
"""
import matplotlib.pyplot as plt
# 交叉验证
from sklearn.model_selection import cross_val_score

def cross_vs(clf, x, y, cv):
    # clf为分类器（还没有训练fit），x,y是验证集，cv是几折交叉验证
    scores = cross_val_score(clf, x, y, cv=cv)
    
    return scores

# K临近法则
from sklearn.neighbors import KNeighborsClassifier

k_range = range(1, 31)
cv_scores = []
# 交叉验证寻找最优参数
# scoring='accuracy' 分类问题使用
# scoring='neg_mean_squared_error'  回归问题使用
def cross_k(n, x_train, y_train, scoring):
    for n in k_range:
        knn = KNeighborsClassifier(n) # 这里是K临近法则
        scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
        #scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='neg_mean_squared_error')
        cv_scores.append(scores.mean())
        
    plt.plot(k_range, cv_scores)
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.show()