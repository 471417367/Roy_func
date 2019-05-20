# 拉格朗日插值法处理NaN值  
from lagrange_nan import lagrange_df, lagrange_df_col  

lagrange_df(df)  
：所有列的NaN进行lagrange进行插值  
lagrange_df_col(df_col)  
：指定列的NaN进行lagrange进行插值  
无返回值  
******************************************************************  

# 属性压缩，归一化和标准化  
from standard_minmax_scaler import standard, minmax  

standard(df)  
：将df中的所有数据标准化，并返回  
minmax(df)  
：将df中的所有数据归一化，并返回  
******************************************************************  

# 划分训练集和测试集  
from train_test_split import train_test  

train_test(df, size)  
:x_train, x_test, y_train, y_test  
size = 0.2 ,表示测试集20%，训练集80%  
******************************************************************  

# 交叉验证  
from cross_val_score import cross_vs  

cross_vs(clf, x, y, cv)  
：clf为分类器（还没有训练fit），x,y是验证集，cv是几折交叉验证。返回一个数组[cv个验证准确率结果]  
cross_k  
：使用此函数需要细看。利用交叉验证找最佳参数  
******************************************************************  

# 可视化混淆矩阵  
from confusion_matrix import cm_plt, cm_plt_net  

cm_plt(clf, x_train, y_train)  
:逻辑回归，SVM，决策树。无返回值，直接出图  
cm_plt_net(net, x_train, y_train)  
：神经元网络模型。无返回值，直接出图  
******************************************************************  

# 神经网络模型  
from kares_model import keras_seq  

keras_seq(input_d, x_train, y_train)  
:返回模型 net  
******************************************************************  

# KMeans聚类模型  
# from k_means import k_means  

k_means(df, k, feature)  
：聚类K个，无返回值  
******************************************************************
