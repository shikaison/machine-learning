import numpy as np
from sklearn.linear_model import LogisticRegression

# 输入数据
x = np.array([9.95, 10.14, 9.22, 8.87, 12.06, 16.30, 17.01, 18.93, 14.01, 13.01, 15.41, 14.21])
y = np.array([1.018, 1.143, 1.036, 0.915, 1.373, 1.640, 1.886, 1.913, 1.521, 1.237, 1.601, 1.496])

# 将问题转化为二元分类问题
y_binary = (y >= 1.5).astype(int)

# 创建并拟合逻辑回归模型
clf = LogisticRegression(solver='liblinear')
clf.fit(x.reshape(-1, 1), y_binary)

# 预测新值
x_new = np.array([13.5]).reshape(-1, 1)
y_pred = clf.predict(x_new)

if y_pred[0] == 1:
    print("对应的y值 >= 1.5")
else:

    print("对应的y值 < 1.5")