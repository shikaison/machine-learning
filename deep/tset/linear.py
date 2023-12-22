import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# 自定义数据集
# X = np.array([207, 187, 187, 170, 194, 240, 257, 184, 257]).reshape(-1, 1)
# Y = np.array([10.4, 9.4, 9.4, 8.7, 11.2, 12.7, 13, 11.2, 13])
# X = np.array([240, 170, 240, 187, 194, 240, 230, 230, 170]).reshape(-1, 1)
# Y = np.array([12.7, 8.7, 12.7, 9.4, 11.2, 12.7, 12.8, 12.8, 8.7])
X = np.array([257, 257, 161, 230, 170, 207, 187, 230, 207]).reshape(-1, 1)
Y = np.array([13, 13, 7.1, 12.8, 8.7, 10.4, 9.4, 12.8, 10.4])
# 初始化线性回归模型
model = LinearRegression()

# 在数据集上拟合线性回归模型
model.fit(X, Y)

# 获取拟合的斜率和截距
slope = model.coef_[0]
intercept = model.intercept_

# 打印拟合结果
print("斜率:", slope)
print("截距:", intercept)
plt.plot(X, Y, 'o', label='Original data', markersize=10)
plt.show()