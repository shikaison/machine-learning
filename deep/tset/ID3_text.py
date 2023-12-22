# -*- coding: utf-8 -*-
"""
Created on Tue Oct 14 22:43:49 2023
@author: HUZHUHUA
"""
##################################
# 决策树-基于 ID3 算法的 Python 实现
##################################
# 导入标准库
import numpy as np

from sklearn.tree import DecisionTreeClassifier


X_train = np.array([[1, 1, 2, 2], [1, 1, 2, 1],
                    [2, 1, 2, 2], [4, 2, 1, 2],
                    [4, 2, 2, 2], [3, 3, 2, 2],
                    [4, 2, 2, 1], [1, 2, 1, 2],
                    [2, 1, 1, 1], [3, 2, 2, 2],
                    [3, 3, 1, 1], [3, 2, 1, 2]])
y_train = np.array([2, 2, 1, 1, 2, 2, 1, 2, 1, 2, 1, 1])
# ID3 分类树，信息增益特征选择
dtree = DecisionTreeClassifier(criterion='entropy')
# 训练
dtree.fit(X_train, y_train)
# 测试
X_test = np.array([[1, 1, 1, 1]])
pred_y = dtree.predict(X_test)
print(pred_y)
# 决策树可视化
import graphviz
from sklearn.tree import export_graphviz

dot_data = export_graphviz(dtree,
                           feature_names=['age', 'income level', 'fixed income', 'VIP'], filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.view()
