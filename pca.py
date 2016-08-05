__author__ = 'ccuulinay'

import numpy as np
from numpy import *


def zeroMean(dataMatrix):
    meanVal = np.mean(dataMatrix, axis=0)  # 按列求均值, 即求各个特征的均值
    newData = dataMatrix - meanVal
    return newData, meanVal


def pca(dataMatrix, topN):
    meanRemoved, meanVal = zeroMean(dataMatrix)

    print shape(meanRemoved)

    # meanRemoved m * n, covMatrix n * n
    # 求协方差矩阵, return ndarray,
    covMatrix = np.cov(meanRemoved, rowvar=0)
    # 从协方差矩阵,求特征值和特征向量
    # 特征值 eigVectors n * n
    eigVals, eigVectors = np.linalg.eig(np.mat(covMatrix))
    # 对特征值从小到大排序
    eigValIndice = np.argsort(eigVals)
    # 最大的topN个特征值的下标
    eigValIndice = eigValIndice[:-(topN+1):-1]
    # Get reorganized eig vectors from eigValIndice
    regEigVectors = eigVectors[:, eigValIndice]
    # Transform data matrix into new dimensions
    # meanRemoved m * n and regEigVectors n * k
    lowDimensionDataMatrix = meanRemoved * regEigVectors
    # lowDimensionDataMatrix m * k
    reconMatrix = (lowDimensionDataMatrix * regEigVectors) + meanVal
    return lowDimensionDataMatrix, reconMatrix


