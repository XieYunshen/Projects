# -*- coding: utf-8 -*-
# @Time    : 2018/11/22 19:46
# @Author  : xieyunshen
# @Email   : xieyunshen_2018@163.com
# @File    : Algorithm4.py
# @Software: PyCharm
# @ModifyTime:
# Unweight Bregman Consensus Clustering With Constraints
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np


def generate_CCM(partitions):
    '''
    construct m cluster aggregation matrix Mi
    :param part: 即partition
    :return: 返回嵌套列表形式的矩阵
    '''
    M_list = []
    for part in partitions:
        n = len(part)
        M = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if part[i] == part[j]:
                    M[i][j] = 1.0
                else:
                    M[i][j] = 0.0
        M_list.append(M)
    return M_list


def UBCCC(partitions, M, C):
    '''
    Unweight Bregman Consensus Clustering With Constraints
    :param partition:
    :param M:(xp,xq) belong to M indicates that xp and xq belong to same cluster
    :param C:(xp,xq) belong to M denotes that xp and xq belong to different cluster
    :return: optimal M
    '''
    M_list = generate_CCM(partitions)
    # m表示partition的数量
    m = len(M_list)
    # n表示矩阵的维度
    n = len(M_list[0])
    sum_ = np.zeros((n, n), dtype=float)
    for cell in M_list:
        sum_ += cell
    optimal_M = sum_/m
    s_max = np.max(optimal_M)
    s_min = np.min(optimal_M)
    if len(M) !=0:
        for cell in M:
            optimal_M[cell[0], cell[1]] = s_max
            optimal_M[cell[1], cell[0]] = s_max
    if len(C) !=0:
        for cell in C:
            optimal_M[cell[0], cell[1]] = s_min
            optimal_M[cell[1], cell[0]] = s_min
    return optimal_M


# 生成M和C集合，M为must-link约束，C为cannot-link约束
def generate_M_C(file):
    data = pd.read_csv(file,delimiter=',',encoding='utf-8',header=0)
    # print(list(data['Death']))
    labels = list(data['Death'])
    # 随机生成20个随机数
    M = []
    C = []
    nums = set(np.random.randint(0, 125, 20))
    # print(nums)
    for i in nums:
        for j in nums:
            if i!=j:
                if labels[i]==labels[j]:
                    M.append([i,j])
                else:
                    C.append([i,j])
    return M,C





# 生成热力图
def generate_heatmap(M,title):
    # cmap = sns.color_palette(flatui)
    # cmap = sns.light_palette("black", reverse=True, n_colors=8)
    f, ax = plt.subplots(figsize=(10, 10))
    cmap = sns.light_palette("black", reverse=True, n_colors=8)
    sns.heatmap(M,cbar=False,cmap=cmap)
    ax.set_title(title)
    plt.show()
    f.savefig('./'+title+'.jpg')


def get_data(file):
    data = pd.read_csv(file,delimiter=',',encoding='utf-8',header=0).round(6)
    title = list(data.ix[:0])
    cluster_data = []
    for i in range(len(title)-1):
        cluster_data.append(list(data.ix[:,title[i]]))
    return cluster_data


def generate_cluster(cluster_data,n_clusters=3):
    X = np.array(cluster_data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    return list(kmeans.labels_)


# 设定不同的K，根据K-means聚类生成partitions。
def generate_partitions(file,s,e):
    cluster_data = get_data(file)
    partitions = []
    for i in range(s,e+1):
        kmeans = generate_cluster(cluster_data,i)
        partitions.append(kmeans)
    return partitions


if __name__=='__main__':

    file_path = '../../data/GBM/Gene.csv'
    partitions = generate_partitions(file_path, 2, 10)
    M, C = generate_M_C('../../data/GBM/Survival.csv')
    # print(M)
    # print(C)
    # exit()
    output_M = UBCCC(partitions, M, C)
    generate_heatmap(output_M,'UBCCC_1')
