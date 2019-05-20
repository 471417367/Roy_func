# -*- coding: utf-8 -*-
"""
@author: Roy
"""
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

# 可视化混淆矩阵:逻辑回归，SVM，决策树
def cm_plt(clf, x_train, y_train):    
    pred = clf.predict(x_train)
    conf_mat = confusion_matrix(y_train, pred) # conf_mat为分类情况的混淆矩阵
    plt.matshow(conf_mat, cmap=plt.cm.Greens)
    plt.colorbar()

    for x in range(len(conf_mat)):
        for y in range(len(conf_mat)):
            plt.annotate(conf_mat[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
        
    plt.title(u'分类器混淆矩阵的可视化展示')
    plt.ylabel('True label');
    plt.xlabel('Predicted label');

# 神经元网络模型
def cm_plt_net(net, x_train, y_train):    
    pred = net.predict_classes(x_train).reshape(len(x_train))
    conf_mat = confusion_matrix(y_train, pred) # conf_mat为分类情况的混淆矩阵
    plt.matshow(conf_mat, cmap=plt.cm.Greens)
    plt.colorbar()

    for x in range(len(conf_mat)):
        for y in range(len(conf_mat)):
            plt.annotate(conf_mat[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
        
    plt.title(u'经元网络模型分类器混淆矩阵的可视化展示')
    plt.ylabel('True label');
    plt.xlabel('Predicted label');