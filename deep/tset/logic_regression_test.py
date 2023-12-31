#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt
# 将根目录添加到 sys.path，解决在命令行下执行时找不到模块的问题
import sys
import os
# 构造伪分类数据并可视化
from sklearn.datasets import make_classification

data, target = make_classification(n_samples=100,
                                   n_features=2, n_classes=2, n_informative=1, n_redundant=0, n_repeated=0,
                                   n_clusters_per_class=1)
print(data.shape)
print(target.shape)
plt.scatter(data[:, 0], data[:, 1], c=target, s=50)
plt.show()
# 训练模型
from my_models.linear_model.logic_regression import LogisticRegression

lr = LogisticRegression()
lr.fit(data, target)
# 查看 loss 值的变化，交叉熵损失
lr.plot_losses()
lr.plot_decision_boundary(data, target)
# 计算 F1
from sklearn.metrics import f1_score

f1_score(target, lr.predict(data))

# 与sklearn 中的逻辑回归对比
# from sklearn.linear_model import LogisticRegression
#
# lr = LogisticRegression()
# lr.fit(data, target)
# w1 = lr.coef_[0][0]
# w2 = lr.coef_[0][1]
# bias = lr.intercept_[0]
# print(w1)
# print(w2)
# print(bias)
# # 画决策边界
# x1 = np.arange(np.min(data), np.max(data), 0.1)
# x2 = -w1 / w2 * x1 - bias / w2
# plt.scatter(data[:, 0], data[:, 1], c=target, s=50)
# plt.plot(x1, x2, 'r')
# plt.show()
# # 计算 F1
# f1_score(target, lr.predict(data))
