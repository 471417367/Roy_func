# �������ղ�ֵ������NaNֵ
from lagrange_nan import lagrange_df, lagrange_df_col

lagrange_df(df)
�������е�NaN����lagrange���в�ֵ
lagrange_df_col(df_col)
��ָ���е�NaN����lagrange���в�ֵ
�޷���ֵ
******************************************************************

# ����ѹ������һ���ͱ�׼��
from standard_minmax_scaler import standard, minmax

standard(df)
����df�е��������ݱ�׼����������
minmax(df)
����df�е��������ݹ�һ����������
******************************************************************

# ����ѵ�����Ͳ��Լ�
from train_test_split import train_test

train_test(df, size)
:x_train, x_test, y_train, y_test
size = 0.2 ,��ʾ���Լ�20%��ѵ����80%
******************************************************************

# ������֤
from cross_val_score import cross_vs

cross_vs(clf, x, y, cv)
��clfΪ����������û��ѵ��fit����x,y����֤����cv�Ǽ��۽�����֤������һ������[cv����֤׼ȷ�ʽ��]
cross_k
��ʹ�ô˺�����Ҫϸ�������ý�����֤����Ѳ���
******************************************************************

# ���ӻ���������
from confusion_matrix import cm_plt, cm_plt_net

cm_plt(clf, x_train, y_train)
:�߼��ع飬SVM�����������޷���ֵ��ֱ�ӳ�ͼ
cm_plt_net(net, x_train, y_train)
����Ԫ����ģ�͡��޷���ֵ��ֱ�ӳ�ͼ
******************************************************************

# ������ģ��
from kares_model import keras_seq

keras_seq(input_d, x_train, y_train)
:����ģ�� net
******************************************************************

# KMeans����ģ��
# from k_means import k_means

k_means(df, k, feature)
������K�����޷���ֵ
******************************************************************








