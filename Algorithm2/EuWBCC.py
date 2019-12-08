# -*- coding: utf-8 -*-
# @Time    : 2018/11/4 14:46
# @Author  : xieyunshen
# @Email   : xieyunshen_2018@163.com
# @File    : Algorithm2.py
# @Software: PyCharm
# @ModifyTime:

# Weight Bregmannian Consensus Clustering
from numpy import *
import numpy as np
import cvxopt
from cvxopt import matrix
import seaborn as sns
import matplotlib.pyplot as plt


def generate_CCM(part):
    '''
    construct m cluster aggregation matrix Mi
    :param part: 即partition
    :return: 返回嵌套列表形式的矩阵
    '''
    n = len(part)
    M = np.zeros((n,n),dtype=float)
    for i in range(n):
        for j in range(n):
            if part[i]==part[j]:
                M[i][j]=1.0
            else:
                M[i][j] = 0.0
    return M


def BregmanDivergence(Mt,Mi):
    '''
    :param Mt:最优CCM矩阵
    :param Mi: partition_i对应的CCM矩阵
    :return: 散度值
    '''
    # 计算函数为欧式距离的布雷格曼散度
    bregman_value = 0.5*Mt**2 - 0.5*Mi**2 - Mi*(Mt-Mi)
    return bregman_value


def get_q(Mt,M_list):
    q = []
    for cell in M_list:
        # r = BregmanDivergence(Mt,cell)
        r = matrix_distance(Mt,cell)
        q.append(r)
        # print(type(r))
    # print(q)
    return q


def function_J2(Mt,wt,M_list,tradeoff):
    m = len(M_list)
    # trad = matrix(tradeoff * np.eye(len(Mt)))
    # sum_ = np.zeros((len(Mt),len(Mt)),dtype=float)
    sum_ = 0.
    for i in range(m):
        sum_ += wt[i]*matrix_distance(Mt,M_list[i])
    result = sum_ + tradeoff*(np.sum(wt**2))
    return result



def EuWBCC(partitons, precision,tradeoff):
    """
    Weight Bregmannian Consensus Clustering using Euclidean Distance
    :param partitons:
    :param precision:
    :param tradeoff:
    :return:
    """
    # 构建M矩阵
    n = len(partitons[0])
    m = len(partitons)
    M_list = []
    for i in range(m):
        M_list.append(generate_CCM(partitons[i]))
    # 初始化
    # 创建行向量
    w0 = np.array([1/m]*m)
    # 创建m维的单位矩阵
    M0 = np.eye(n)
    t = 0
    delta = float('inf')
    last_J2 = function_J2(M0,w0,M_list,tradeoff)
    Mt = M0
    # f = open('./w_value.txt','a',encoding='utf-8')
    # f.write('='*40+'\n')
    wt = w0
    while delta > precision:
        t+=1
        # 计算Mt
        Mt = compute_Mt(M_list,wt)
        # 计算wt的值
        p = matrix(2*tradeoff*np.eye(m))
        q_l = np.array(get_q(Mt, M_list))
        q = matrix(q_l, (m,1))
        G = matrix(-1*np.eye(m))
        h = matrix(0., (m,1))
        A = matrix(1.,(1,m))
        b = matrix(1.)
        wt = cvxopt.solvers.qp(P=p,q=q,G=G,h=h,A=A,b=b)['x']
        # f.write(str(wt)+'\n')
        this_J2 = function_J2(Mt,wt,M_list,tradeoff)
        delta = abs(this_J2 - last_J2)
        last_J2 = this_J2
    return Mt


# 计算矩阵的欧式距离
def matrix_distance(X1,X2):
    C = X1-X2
    C2 = multiply(C,C)
    D=sqrt(sum(C2[:]))
    return D

def compute_Mt(Mt_list,w):
    m = len(w)
    n = len(Mt_list[0][0])
    sum_ = np.zeros((n, n), dtype=float)
    for i in range(m):
        sum_ += w[i]*mat(Mt_list[i])
    return sum_


if __name__ =='__main__':
    # Mirna
    p1 = [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 2, 1, 0, 1, 0, 0, 2, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 2, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 2, 0, 2, 2, 1, 0, 1, 2, 0, 1, 2, 2, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 2, 0, 1, 1, 1, 1, 1, 0, 1, 1, 2, 0, 0, 1, 0, 0, 1, 2, 1, 0, 0, 1, 1, 0, 0, 0, 0, 2, 0, 0, 1, 0, 2, 0, 2, 0, 2, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 1, 1, 0, 2, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 2, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0]
    # Gene
    p2 = [2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0]
    # Methy
    p3 = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 2, 0, 2, 0, 2, 2, 1, 2, 1, 0, 2, 1, 1, 1, 2, 0, 2, 0, 2, 2, 2, 0, 1, 1, 0, 2, 2, 2, 2, 1, 1, 1, 0, 1, 2, 2, 1, 1, 2, 1, 2, 2, 2, 1, 0, 1, 2, 0, 2, 2, 2, 2, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 2, 2, 1, 0, 1, 2, 0, 2, 0, 2, 1, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 0, 2, 2, 2, 0, 0, 2, 2, 2, 0]
    partitions = [p1,p2,p3]
    M = EuWBCC(partitons=partitions, precision=0.1, tradeoff=8)
    # generate_heatmap(M, 'consensusKmeans')

    # a = np.eye(3)
    # b = 2* np.eye(3)
    # c = np.array([a,b])
    # print(c.shape)