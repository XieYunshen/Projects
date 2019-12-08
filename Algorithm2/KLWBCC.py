# -*- coding: utf-8 -*-
# @Time    : 2018/11/28 16:35
# @Author  : xieyunshen
# @Email   : xieyunshen_2018@163.com
# @File    : KLWBCC.py
# @Software: PyCharm
# @ModifyTime:
import cvxopt
from util import *
import numpy as np
from cvxopt import matrix
import pandas as pd


def get_q(Mt,M_list):
    q = []
    for cell in M_list:
        r = KLdivergence_new(Mt, cell)
        q.append(r)
    return q


def function_J2(Mt,wt,M_list,tradeoff):
    m = len(M_list)
    # trad = matrix(tradeoff * np.eye(len(Mt)))
    # sum_ = np.zeros((len(Mt),len(Mt)),dtype=float)
    sum_ = 0.
    for i in range(m):
        sum_ += wt[i]*KLdivergence_new(Mt,M_list[i])
    result = sum_ + tradeoff*(np.sum(wt**2))
    return result


def KLWBCC(partitons, precision,tradeoff):
    """
    Weight Bregmannian Consensus Clustering using Kullback-Leibler Distance
    :param partitons:
    :param precision:
    :param tradeoff:
    :return:
    """
    # 矩阵的维度n*n
    n = len(partitons[0])
    # partition的数量
    m = len(partitons)
    M_list = []
    for i in range(m):
        M_list.append(generate_CCM(partitons[i]))
    # 初始化
    # 创建行向量
    w0 = np.array([1/m]*m)
    # 创建m维的单位矩阵
    M0 = np.eye(n)
    # t = 0
    # delta表示无穷大
    delta = float('inf')
    last_J2 = function_J2(M0,w0,M_list,tradeoff)
    Mt = M0
    wt = w0
    trend = []
    while delta > precision:
        # t+=1
        # # 计算Mt
        # sum_ = np.zeros((n,n),dtype=float)
        # for i in range(m):
        #     sum_ += wt[i]*np.log(M_list[i])
        # # 得出Mt的值
        Mt = compute_Mt(M_list,wt)
        # 计算wt的值
        trend.append(list(wt.T))
        # print('-'*10)
        # print('trend',trend)
        p = matrix(2*tradeoff*np.eye(m))
        q_l = np.array(get_q(Mt, M_list))
        q = matrix(q_l, (m,1))
        G = matrix(-1*np.eye(m))
        h = matrix(0., (m,1))
        A = matrix(1.,(1,m))
        b = matrix(1.)
        wt = cvxopt.solvers.qp(P=p,q=q,G=G,h=h,A=A,b=b)['x']
        this_J2 = function_J2(Mt,wt,M_list,tradeoff)
        delta = abs(this_J2 - last_J2)
        last_J2 = this_J2
    # print(trend)
    # np.savetxt('KLWBCC_weight_trend.txt', np.array(trend), delimiter=',')
    return Mt


def Accuracy(class_labels,result):
    N = len(class_labels)
    same = 0
    different = 0
    for i in range(N):
        if class_labels[i] == result[i]:
            same += 1
        else:
            different += 1
    acc = max(same, different) / N
    return acc


# 获取共识聚类结果的标签
def WBCC_result(M):
    # 因为结果是二分类，所以创建两个集合，存储样本的序号
    l1 = []
    for i in range(len(M)):
        l1.append(M[:,i])
    X = np.array(l1)
    kmeans = KMeans(n_clusters=2).fit(X)
    return list(kmeans.labels_)


def get_class_label(file):
    data = pd.read_csv(file, delimiter=',', encoding='utf-8', header=0)
    # print(list(data['Death']))
    labels = list(data['Death'])
    return labels


def compute_Mt(Mt_list, w):
    m = len(w)
    n = len(Mt_list[0][0])
    # sum_ = np.zeros((n, n), dtype=float)
    sum_ = np.array(w[0] * mat(np.log(Mt_list[0])))
    for i in range(1, m):
        sum_ *= np.array(w[i] * mat(np.log(Mt_list[i])))

    return np.exp(sum_)