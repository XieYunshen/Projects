# -*- coding: utf-8 -*-
# @Time    : 2018/11/2 17:22
# @Author  : xieyunshen
# @Email   : xieyunshen_2018@163.com
# @File    : Algorithm1.py
# @Software: PyCharm
# @ModifyTime:
# Algorithm 1 Unweighted Bregmannian Consensus Clustering
from numpy import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def EuUBCC(partitions):
    '''
    Unweighted Bregmannian Consensus Clustering using Euclidean Distance
    :param partitions: 聚类的结果
    :return:
    '''
    # 对于多个partition分别求其M矩阵,以list的格式存储在Mi中
    Mi = []
    for cell in partitions:
        n = len(cell)  # 矩阵的维度n
        M = [([0] * n) for i in range(n)]
        for i in range(n):
            for j in range(n):
                if cell[i] == cell[j]:
                    M[i][j]=1
                else:
                    M[i][j]=0
        Mi.append(M)
    # 对矩阵求和
    sum_ = mat([([0] * n) for i in range(n)])
    for cell in Mi:
        sum_ += mat(cell)
    optimal_M = sum_ / len(partitions)
    return optimal_M


def ExpUBCC(partitions):
    '''
    Unweighted Bregmannian Consensus Clustering using Exponential Distance
    :param partitions: 聚类的结果
    :return:
    '''
    # 对于多个partition分别求其M矩阵,以list的格式存储在Mi中
    Mi = []
    for cell in partitions:
        n = len(cell)  # 矩阵的维度n
        M = [([0] * n) for i in range(n)]
        for i in range(n):
            for j in range(n):
                if cell[i] == cell[j]:
                    M[i][j] = 1
                else:
                    M[i][j] = 0
        Mi.append(M)

    # 对矩阵求和
    sum_ = mat([([0.0] * n) for i in range(n)])
    for cell in Mi:
        sum_ += np.exp(mat(cell))
    optimal_M = sum_ / len(partitions)
    return optimal_M


def generate_CCM(part):
    '''
    :param part: 即partition
    :return: 返回嵌套列表形式的矩阵
    '''
    n = len(part)
    M = [([0] * n) for i in range(n)]
    for i in range(n):
        for j in range(n):
            if part[i]==part[j]:
                M[i][j]=1
            else:
                M[i][j] = 0
    return M


def generate_heatmap(M,title):
    # cmap = sns.color_palette(flatui)
    cmap = sns.light_palette("black", reverse=True, n_colors=8)
    f, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(M,cbar=False,cmap=cmap)
    ax.set_title(title)
    f.savefig('./'+title+'.jpg')
    # plt.show()


if __name__ == '__main__':
    # Mirna
    p1 = [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 2, 1, 0, 1, 0, 0, 2, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 2, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 2, 0, 2, 2, 1, 0, 1, 2, 0, 1, 2, 2, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 2, 0, 1, 1, 1, 1, 1, 0, 1, 1, 2, 0, 0, 1, 0, 0, 1, 2, 1, 0, 0, 1, 1, 0, 0, 0, 0, 2, 0, 0, 1, 0, 2, 0, 2, 0, 2, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 1, 1, 0, 2, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 2, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0]
    # Gene
    p2 = [2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0]
    # Methy
    p3 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 2, 0, 2, 0, 2, 2, 1, 2, 1, 0, 2, 1, 1, 1, 2, 0, 2, 0, 2, 2, 2, 0, 1, 1, 0, 2, 2, 2, 2, 1, 1, 1, 0, 1, 2, 2, 1, 1, 2, 1, 2, 2, 2, 1, 0, 1, 2, 0, 2, 2, 2, 2, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 2, 2, 1, 0, 1, 2, 0, 2, 0, 2, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 0, 2, 2, 2, 0, 0, 2, 2, 2, 0]
    partitions = [p1,p2,p3]
    optimal_M = EuUBCC(partitions)

    # generate_heatmap(optimal_M, 'consensusKmeans')
    # M1 = generate_CCM(p1)
    # M2 = generate_CCM(p2)
    # M3 = generate_CCM(p3)
    # generate_heatmap(M1, 'Mirna')
    # generate_heatmap(M2, 'Gene')
    # generate_heatmap(M3, 'Methy')
