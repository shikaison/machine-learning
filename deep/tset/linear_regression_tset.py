#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
###将根目录添加到 sys.path，解决在命令行下执行时找不到模块的问题
import sys
import os

# from my_models.linear_model import LinearRegression

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
# # 造伪样本
# X = np.linspace(0, 100, 100)
X = [207, 187, 187, 170, 194, 240, 257, 184, 257]
w = np.asarray([3, 2])  # 参数
X = np.c_[X, np.ones(9)]  # 考虑到偏置 b
Y = X.dot(w)
Y = [10.4, 9.4, 9.4, 8.7, 11.2, 12.7, 13, 11.2, 13]
X = X.astype('float')
Y = Y.astype('float')
X[:, 0] += np.random.normal(size=X[:, 0].shape) * 3  # 添加 0,1 高斯噪声

Y = Y.reshape(9, 1)

# X = [207, 187, 187, 170, 194, 240, 257, 184, 257]
# X = np.c_[X, np.ones(100)]  # 考虑到偏置 b
# Y = [10.4, 9.4, 9.4, 8.7, 11.2, 12.7, 13, 11.2, 13]
# 与 sklearn 对比
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X, Y)
# predict = lr.predict(X[:, :-1])
# 查看 w,b
print('w:', lr.coef_, 'b:', lr.intercept_)
# 查看标准差
# print(np.std(Y - predict))
# 可视化结果
plt.scatter(X[:, 0], Y)
# plt.plot(X[:, 0], predict, 'r')
# plt.plot(np.arange(0, 100).reshape((100, 1)), Y, c='b', linestyle='--')
plt.show()
