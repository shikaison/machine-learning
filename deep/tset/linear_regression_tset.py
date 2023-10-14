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
# 造伪样本
X = np.linspace(0, 100, 100)
X = np.c_[X, np.ones(100)]  # 考虑到偏置 b
w = np.asarray([3, 2])  # 参数
Y = X.dot(w)
X = X.astype('float')
Y = Y.astype('float')
X[:, 0] += np.random.normal(size=X[:, 0].shape) * 3  # 添加 0,1 高斯噪声
Y = Y.reshape(100, 1)
from my_models.linear_model.linear_regression import *
# 测试
# lr = LinearRegression(solver='sgd')
lr = LinearRegression(solver='closed_form')
lr.fit(X[:, :-1], Y)
predict = lr.predict(X[:, :-1])
# 查看 w
print('w', lr.get_params())
# 查看标准差，如果标准差小的话则认为真实值与预测值相符合。
print(np.std(Y - predict))
# 可视化结果
lr.plot_fit_boundary(X[:, :-1], Y)  # 预测的拟合直线
plt.plot(np.arange(0, 100).reshape((100, 1)), Y, c='b', linestyle='--')
# 真实直线
plt.show()  # 可视化显示
