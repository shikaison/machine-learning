# -*- coding: utf-8 -*-
# 导入标准库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Kmeans(object):
    def __init__(self):
        pass

    # 定义函数：计算欧式距离
    def euclDistance(self, point1, point2):
        # 计算两点 point1、 point2 之间的欧式距离
        distance = np.sqrt(sum(pow(point2 - point1, 2)))
        return distance

    # 定义函数：初始化质心
    def initCentroids(self, dataSet, k):
        # dataSet 为数据集
        # k 是指用户设定的 k 个簇
        numSamples, dim = dataSet.shape  # numSample: 数据集数量； dim: 特征维度
        centroids = np.zeros((k, dim))  # 存放质心坐标，初始化 k 行、 dim 列零矩阵
        for i in range(k):
            # index = int(np.random.uniform(0, numSamples)) # 给出一个服从均匀分布的在0~numSamples 之间的整数
            index = np.random.randint(0, numSamples)  # 给出一个随机分布在 0~numSamples之间的整数
            centroids[i, :] = dataSet[index, :]  # 第 index 行作为质心
        return centroids

    # 定义函数： K-means 聚类
    def kmeans(self, dataSet, k):
        # dataSet 为数据集
        # k 是指用户设定的 k 个簇
        numSamples = dataSet.shape[0]
        clusterAssment = np.zeros((numSamples, 2))  # clusterAssment 第 1 列存放所属的簇，第 2列存放与质心的距离
        clusterChanged = True  # clusterChanged=False 时迭代更新终止
        ## step 1: 初始化质心 centroids
        centroids = self.initCentroids(dataSet, k)
        # 循环体：是否更新质心
        while clusterChanged:
            clusterChanged = False  # 关闭更新
            # 对每个样本点
            for i in range(numSamples):
                minDist = 100000.0  # 最小距离
                minIndex = 0  # 最小距离对应的簇
                ## step2: 找到距离每个样本点最近的质心
                # 对每个质心
                for j in range(k):
                    distance = self.euclDistance(centroids[j, :], dataSet[i, :])  # 计算每个样本点到质心的欧式距离
                    if distance < minDist:  # 如果距离小于当前最小距离 minDist
                        minDist = distance  # 最小距离更新
                        minIndex = j  # 样本所属的簇也会更新
                ## step 3: 更新样本所属的簇
                if clusterAssment[i, 0] != minIndex:  # 如当前样本不属于该簇
                    clusterChanged = True  # 聚类操作需要继续
                clusterAssment[i, :] = minIndex, minDist
            ## step 4: 更新质心
            # 对每个质心
            for j in range(k):
                pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0] == j)[0]]
                # pointsInCluster 存储的是当前所有属于簇 j 的 dataSet 样本点
                centroids[j, :] = np.mean(pointsInCluster, axis=0)  # 更新簇 j 的质心
        print("cluster complete!")
        return centroids, clusterAssment

    def selectK(self, dataSet, k_list):
        # k_list 不同 k 值列表
        distanceK = []  # 存储不同 k 值下每个样本点到质心的平均欧式距离
        for i, k in enumerate(k_list):
            centroids, clusterAssment = self.kmeans(dataSet, k)  # 调用 kmeans 函数
            distance = np.mean(clusterAssment[:, 1], axis=0)  # clusterAssment 所有 minDist 的平均值
            distanceK.append(distance)
        best_k = k_list[np.argmin(distanceK)]  # 能够让距离最小的 k 值
        return distanceK, best_k

    def showCluster(self, dataSet, k, centroids, clusterAssment):
        # dataSet 为数据集
        # k 是指用户设定的 k 个簇
        # centroids 存放质心坐标
        # clusterAssment 第 1 列存放所属的簇，第 2 列存放与质心的距离
        numSamples, dim = dataSet.shape  # numSample: 数据集数量； dim: 特征维度
        if dim != 2:
            print("The dimension of data is not 2!")
            return 1
        mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
        if k > len(mark):
            print("K is too large!")
            return 1
        # 画所有的样本
        plt.figure()
        for i in range(numSamples):
            markIndex = int(clusterAssment[i, 0])
            plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])
        mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
        # 画所有的质心
        for i in range(k):
            plt.plot(centroids[i, 0], centroids[i, 1], mark[i], ms=12.0)
        plt.title('K-means(K={})'.format(k))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
