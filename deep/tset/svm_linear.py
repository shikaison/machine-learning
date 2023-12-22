# -*- coding: utf-8 -*-
# 导入标准库
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# 导入自己定义的数据集
X = np.array([[-1, 1], [0, 0], [0, 3], [1, 2], [2, 2], [2, 0], [3, 0], [2, -2], [1, -3], [3, -2]])
y = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
# 训练-线性支持向量机
clf = svm.SVC(kernel='linear')
clf.fit(X, y)

# 通过查文档获取参数 w
w = clf.coef_[0]

# 通过查文档获取参数 b
b = clf.intercept_[0]

# 计算分类超平面的斜率
w0 = w[0]
w1 = w[1]
b = b
x1 = np.linspace(-2, 4)
y1 = (-w0 * x1 - b) / w1

# 从-2 到 4，顺序间隔采样 50 个样本，默认 num=50， 用于画图
xx = np.linspace(-2, 4)
# 得到分类超平面 w_0*x_1 + w_1*x_2 = 0
y_mid = (w0 * xx + b) / (-w1)
y_up = (w0 * xx + b + 1) / (-w1)
y_down = (w0 * xx + b - 1) / (-w1)

# 获得支持向量对应的两条直线 y_up 和 y_down
# 1) 首先打印出支持向量
print(clf.support_vectors_)
# 2) 得到 y_up 和 y_down
y_up = (w0 * xx + b + 1) / (-w1)
y_down = (w0 * xx + b - 1) / (-w1)

# 画出原数据分布&支持向量
plt.scatter(X[:, 0], X[:, 1], c='b')
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, c='r')
plt.axis('tight')
# 画图：分类超平面、支持向量对应的两条直线
plt.plot(xx, y_mid, 'k-', label='y_mid')
plt.plot(xx, y_up, 'r--', label='y_up')
plt.plot(xx, y_down, 'r--', label='y_down')
plt.legend()
plt.show()
# 计算拉格朗日乘子
a_y = clf.dual_coef_  # a_y = alpha_n * y_n
sv_y = y[clf.support_]  # 支持向量的标签
a = a_y / sv_y  # 拉格朗日因子
print("a_n = {}".format(a))
# 对训练集进行预测， 因为是硬间隔 SVM， 所以精度是 100%
from sklearn.metrics import accuracy_score

pred = clf.predict(X)
accuracy = accuracy_score(y, pred)
print("Accuracy in TrainDataset is: {}".format(accuracy))

