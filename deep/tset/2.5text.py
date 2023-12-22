import numpy as np

# S = [110, 140, 142.5, 155, 160, 170, 177, 187.5, 235, 245]
# P = [199, 245, 319, 240, 312, 279, 310, 308, 405, 324]
#
X = np.array([1, 2, 3, 4, 5, 1, 1, 1, 1, 1], dtype=float)  # 输入特征，将其转换为浮点数类型
y = np.array([2, 4, 5, 4, 5, 1, 1, 1, 1, 1], dtype=float)  # 实际输出，将其转换为浮点数类型
# for i in range(len(S)):
#     X[i] = "{:.2f}".format((S[i] - min(S)) / (max(S) - min(S)))
#     y[i] = "{:.2f}".format((P[i] - min(P)) / (max(P) - min(P)))
#     print(X[i], y[i])

S = [0.00, 0.22, 0.24, 0.33, 0.37, 0.44, 0.44, 0.57, 0.93, 1.00]

P = [0.11, 0.22, 0.58, 0.20, 0.55, 0.39, 0.54, 0.53, 1.00, 0.61]
for i in range(10):
    X[i] = S[i]
    y[i] = P[i]
    print(X[i], y[i])

# 生成一些示例数据


# 初始化模型参数
theta0 = 1.0  # 初始化权重
theta1 = 1.0  # 初始化偏差

# 学习率
alpha = 0.01

# 迭代次数
num_iterations = 1000
predictions = np.array([2, 4, 5, 4, 5, 1, 1, 1, 1, 1], dtype=float)  # 实际输出，将其转换为浮点数类型
# 梯度下降优化
for i in range(num_iterations):
    # 计算预测值
    for n in range(10):
        predictions[n] = "{:.2f}".format(theta0 + theta1 * X[n])
    # predictions = theta0 + theta1 * X
    # 计算损失函数的梯度
    gradient_theta0 = np.round((-1) * np.sum(y - predictions), 6)
    gradient_theta1 = np.round((-1) * np.sum((y - predictions) * X), 6)
    print("第", i, "次迭代的梯度为(gradient_theta0):", gradient_theta0)
    print("第", i, "次迭代的梯度为(gradient_theta1):", gradient_theta1)
    # 使用梯度下降更新参数
    theta0 = theta0 - alpha * gradient_theta0
    theta1 = theta1 - alpha * gradient_theta1
    print("第", i, "次迭代 (theta0):", theta0)
    print("第", i, "次迭代 (theta1):", theta1)
# 输出最优参数
print("最优权重 (theta0):", theta0)
print("最优权重 (theta1):", theta1)
