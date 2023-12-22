import numpy as np

# 原始数据集
original_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

# 自助采样的次数
num_iterations = 3

# 存储每次采样的结果
bootstrap_samples = []

for i in range(num_iterations):
    # 从原始数据中有放回地抽样相同数量的数据
    bootstrap_sample = np.random.choice(original_data, size=len(original_data), replace=True)
    bootstrap_samples.append(bootstrap_sample)

# 打印每次采样的结果
for i, sample in enumerate(bootstrap_samples):
    print(f"Iteration {i + 1}: {sample}")
