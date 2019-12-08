# -*- coding: utf-8 -*-
# @Time    : 2019/1/26 17:17
# @Author  : xieyunshen
# @Email   : xieyunshen_2018@163.com
# @File    : SSKLWBCC.py
# @Software: PyCharm
# @ModifyTime:
# Weighted Bregmannian Consensus Clustering With Constraints

from util import *
import cvxopt
from cvxopt import matrix



def SSKLWBCC(partitions,M,C,precision,tradeoff):
    """
    Weighted Bregmannian Consensus Clustering With Constraints using Kullback-Leibler Distance
    :param partitions:
    :param M: M is the must-link constraints
    :param C: C is the cannot-link constraints
    :param precision:
    :param tradeoff:
    :return:
    """
    M_list = generate_CCM_list(partitions)
    # m表示partition的数量
    m = len(M_list)
    # n表示矩阵的维度
    n = len(M_list[0])
    w0 = np.array([1/m]*m)
    M0 = np.eye(n)
    last_J2 = function_J2(M0,w0,M_list,tradeoff)
    Mt = M0
    delta = float('inf')
    wt = w0
    while delta > precision:
        # 计算Mt
        # sum_ = np.zeros((n,n),dtype=float)
        # for i in range(m):
        #     sum_ += wt[i]*mat(M_list[i])
        # # 得出Mt的值
        Mt = compute_Mt(M_list,wt)
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
    # trad = matrix(tradeoff * np.eye(len(Mt)))
    # sum_ = np.zeros((len(Mt),len(Mt)),dtype=float)
    sum_ = 0.
    for i in range(m):
        sum_ += wt[i]*KLdivergence_new(Mt,M_list[i])
    result = sum_ + tradeoff*(np.sum(wt**2))
    return result


def get_q(Mt,M_list):
    q = []
    for cell in M_list:
        r = KLdivergence_new(Mt,cell)
        q.append(r)
    return q


def compute_Mt(Mt_list, w):
    m = len(w)
    n = len(Mt_list[0][0])
    # sum_ = np.zeros((n, n), dtype=float)
    sum_ = np.exp(np.array(w[0] * mat(np.log(Mt_list[0]))))
    for i in range(1, m):
        sum_ *= np.exp(np.array(w[i] * mat(np.log(Mt_list[i]))))

    return sum_