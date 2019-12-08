# -*- coding: utf-8 -*-
# @Time    : 2019/1/14 19:11
# @Author  : xieyunshen
# @Email   : xieyunshen_2018@163.com
# @File    : SSEuWBCC.py
# @Software: PyCharm
# @ModifyTime:

# Weighted Bregmannian Consensus Clustering With Constraints
from numpy import *
import numpy as np
import cvxopt
from cvxopt import matrix



def SSEuWBCC(partitions,M,C,precision,tradeoff):
    """
    Weighted Bregmannian Consensus Clustering With Constraints using Euclidean Distance
    :param partitions:
    :param M: M is the must-link constraints
    :param C: C is cannot-link constraints
    :param precision:
    :param tradeoff:
    :return:
    """
    M_list = generate_CCM(partitions)
    # m表示partition的数量
    m = len(M_list)
    # n表示矩阵的维度
    n = len(M_list[0])
    w0 = np.array([1/m]*m)
    M0 = np.eye(n)
    t = 0
    last_J2 = function_J2(M0,w0,M_list,tradeoff)
    Mt = M0
    delta = float('inf')
    wt = w0
    while delta > precision:
        # 计算Mt
        sum_ = np.zeros((n,n),dtype=float)
        for i in range(m):
            sum_ += wt[i]*mat(M_list[i])
        # 得出Mt的值
        Mt = sum_
        s_max = np.max(Mt)
        s_min = np.min(Mt)
        if len(M) !=0:
            for cell in M:
                Mt[cell[0], cell[1]] = s_max
                Mt[cell[1], cell[0]] = s_max
        if len(C) !=0:
            for cell in C:
                Mt[cell[0], cell[1]] = s_min
                Mt[cell[1], cell[0]] = s_min

        # obstain wt
        p = matrix(2 * tradeoff * np.eye(m))
        q_l = np.array(get_q(Mt, M_list))
        q = matrix(q_l,(m,1))
        G = matrix(-1 * np.eye(m))
        h = matrix(0., (m, 1))
        A = matrix(1., (1, m))
        b = matrix(1.)
        wt = cvxopt.solvers.qp(P=p, q=q, G=G, h=h, A=A, b=b)['x']
        this_J2 = function_J2(Mt, wt, M_list, tradeoff)
        delta = abs(this_J2 - last_J2)
        last_J2 = this_J2
    return Mt


def function_J2(Mt,wt,M_list,tradeoff):
    m = len(M_list)
    sum_ = 0.
    for i in range(m):
        sum_ += wt[i]*matrix_distance(Mt,M_list[i])
    result = sum_ + tradeoff*(np.sum(wt**2))
    return result


# 计算矩阵的欧式距离
def matrix_distance(X1,X2):
    C = X1-X2
    C2 = multiply(C,C)
    D=sqrt(sum(C2[:]))
    return D


def generate_CCM(partitions):
    '''
    construct m cluster aggregation matrix Mi
    :param partitions: 即partition
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





def get_q(Mt,M_list):
    q = []
    for cell in M_list:
        r = float(matrix_distance(Mt,cell))
        q.append(r)
    return q
