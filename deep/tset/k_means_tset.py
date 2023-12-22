# -*-coding:utf-8-*-

import matplotlib.pyplot as plt
# 导入标准库
import pandas as pd
from my_models.k_means.k_means import Kmeans

# 导入数据集
means = Kmeans()
dataSet = pd.read_csv('dataSet.csv')
dataSet = dataSet.values  # (80,2)
# 选择不同的 k 值对比
k_list = [2, 3, 4, 5, 6]
disK, best = means.selectK(dataSet, k_list)
# 画图
print('best=', best)
plt.figure()
plt.plot(k_list, disK, 'ro-')
plt.title('Cross-validation on k')
plt.xlabel('K')
plt.ylabel('Mean Euclidean distance')
plt.show()
# 使用 K-means 算法进行聚类
k = 4
centroids, clusterAssment = means.kmeans(dataSet, 4)
# 作图可视化 kmeans 聚类效果
means.showCluster(dataSet, k, centroids, clusterAssment)
