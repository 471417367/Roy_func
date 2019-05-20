# -*- coding: utf-8 -*-
"""
@author: Roy
"""
# 神经网络模型
from keras.models import Sequential
from keras.layers.core import Dense, Activation

def keras_seq(input_d, output_d = 10, x_train, y_train):
    net = Sequential()
    # 添加输入层（input_d节点）到隐藏层（output_d节点）的连接
    # input_d是x的属性数量，output_d可以设置为10
    net.add(Dense(input_dim=input_d, output_dim=output_d))
    net.add(Activation('relu'))  # 隐藏层使用relu激活函数
    net.add(Dense(input_dim=output_d, output_dim=1))   # 添加隐藏层到输出层的连接
    net.add(Activation('sigmoid'))  # 输出层使用sigmoid激活函数
    net.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # 编译模型，使用adam方法求解

    net.fit(x_train, y_train, nb_epoch=100, batch_size=1)
    
    return net
