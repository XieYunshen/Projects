# -*- coding: utf-8 -*-
# @Time    : 2018/11/26 17:30
# @Author  : xieyunshen
# @Email   : xieyunshen_2018@163.com
# @File    : util.py
# @Software: PyCharm
# @ModifyTime:
from numpy import *
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D
import random
from scipy import stats


# 计算矩阵的欧式距离
def Euclidean_distance(X1,X2):
    C = X1-X2
    C2 = multiply(C,C)
    D=sqrt(sum(C2[:]))
    return D


# 计算矩阵的欧式距离方法二
def matrix_distance1(X1,X2):
    C = X1 - X2
    # print(C)
    D = dot(C,C)
    E = trace(D)
    return E**0.5


# 欧氏距离的平方
def matrix_distance2(X1,X2):
    C = X1 - X2
    D = np.linalg.norm(C, ord=2)
    return D**2


# 生成热力图
def generate_heatmap(M,title):
    # cmap = sns.color_palette(flatui)
    # cmap = sns.light_palette("black", reverse=True, n_colors=8)
    f, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(M,cbar=False,xticklabels=False,yticklabels=False)
    ax.set_title(title)
    f.savefig('./'+title+'.png')
    # plt.show()


# 生成CCM矩阵
def generate_CCM(part):
    '''
    construct m cluster aggregation matrix Mi
    :param part: 即partition
    :return: 返回嵌套列表形式的矩阵
    '''
    n = len(part)
    # print('len(part)',n)
    M = np.zeros((n,n),dtype=float)
    for i in range(n):
        for j in range(n):
            if part[i]==part[j]:
                M[i][j]=1.0
            else:
                M[i][j] = 0.00000001
    return M


# Kmeans聚类后，返回标签。
def generate_cluster(cluster_data,n_clusters=2):
    X = np.array(cluster_data)
    kmeans = KMeans(n_clusters=n_clusters).fit(X)
    return list(kmeans.labels_)


# 设定不同的K，根据K-means聚类生成partitions。
def generate_partitions(file,s,e):
    cluster_data = get_data(file)
    partitions = []
    for i in range(s,e+1):
        kmeans = generate_cluster(cluster_data,i)
        partitions.append(kmeans)
    return partitions


# 读取CSV文件
def get_data(file):
    '''
    :param file:获取数据集中的实例的特征向量
    :return: 返回嵌套列表，每一个子列表代表一个实例的特征向量
    '''
    data = pd.read_csv(file, delimiter=',',encoding='utf-8',header=0).round(6)
    title = list(data.ix[:0])
    # for cell in title:
    #     print(cell)
    # print(len(title))
    cluster_data = []
    for i in range(len(title)-1):
        cluster_data.append(list(data.ix[:,title[i]]))
        # print(list(data.ix[:,title[i]]))
    return cluster_data


# KL散度的计算
def KLDivergence(X1,X2):
    I = len(X1)
    J = len(X2)
    sum_ = 0.0
    for i in range(I):
        for j in range(J):
            ll = X1[i][j]/X2[i][j]
            # try:
            #     ll = X1[i][j]/X2[i][j]
            # except ZeroDivisionError:
            #     ll = X1[i][j]/0.000001
            try:
                sum_ += X1[i][j] * log(ll) - X1[i][j] + X2[i][j]
            except RuntimeWarning:
                sum_ += X2[i][j] - X1[i][j]

    return sum_


def KLdivergence_new(X,Y):
    arr_X = np.array(X).flatten()
    arr_Y = np.array(Y).flatten()
    vector_x = arr_X/np.max(arr_X)
    vector_y = arr_Y/np.max(arr_Y)
    distance = stats.entropy(vector_x,vector_y)

    print('KLdivergence:',distance)
    return distance

# 指数距离的计算
def expDistance(X1,X2):
    distance = np.exp(X1) - np.exp(X2) - (X1-X2)*np.exp(X2)
    r = np.linalg.norm(distance, ord=2)
    return r


# 指数距离的计算
def expDistance_new(X,Y):
    vector_x = np.array(X).flatten()
    vector_y = np.array(Y).flatten()
    distance = np.exp(vector_x) - np.exp(vector_y) - (vector_x-vector_y)*np.exp(vector_y)
    r = math.log(np.linalg.norm(distance, ord=2))
    print('ExpDistance:',r)
    return r


# 场向量
def Fieldervector(M):
    # D_list = []
    m = len(M)
    # print(M.sum(axis=1))
    D_ = map(sum,M)
    D = matrix(np.diag(list(D_)))
    # print(D)
    D1 = np.sqrt(D).I
    # print(D1)
    I = np.eye(m)
    L = I - D1*M*D1
    # print(L)
    eigenvalue,eigenvector = np.linalg.eig(L)
    # print(eigenvalue)
    # print(eigenvector)
    s = second_min(eigenvalue)
    # print(s)
    result = eigenvector[:,s].tolist()
    # print(result)
    # result = eigenvector.T.tolist()[s]
    # print(result.shape())
    return result


# 生成Eigenvector图像
def Eigenvector_image(x_y,title):
    # print(len(x_y))
    plt.figure()
    x = list(range(0, len(x_y)))
    y = x_y
    plt.scatter(x, y, label="Eigenvector",s=1)
    plt.title(title)
    plt.legend()
    plt.savefig(title+'.png')
    # plt.show()


# 获取列表中倒数第二的元素序号,这里如果最小的几个特征值为0，则次小的特征值应该为大于零的最小数
def second_min(lt):
    d = {}
    for i,v in enumerate(lt):
        d[v] = i
    # print(lt)
    index = list(set(lt))
    index.sort()
    y = index[1]
    return d[y]


# 获取文件的标签
def get_class_label(file):
    data = pd.read_csv(file, delimiter=',', encoding='utf-8', header=0)
    # print(list(data['Death']))
    labels = list(data['Death'])
    return labels


# 原始数据降维后可视化
def data_visualization_2D(cluster_data, class_labels,title):
    # class_labels = get_class_label(labelfile)
    # 源数据可视化
    # cluster_data = get_data(file)
    n_components = 2
    X = np.array(cluster_data)
    tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
    # tsne = manifold.TSNE(n_components=n_components, init='pca')
    Y = tsne.fit_transform(X)  # 转换后的输出
    fig = plt.figure()
    axes = fig.add_subplot(111)

    for i in range(len(cluster_data)):
        if class_labels[i] == 0:
            axes.scatter(Y[i, 0], Y[i, 1], color='red')
        if class_labels[i] == 1:
            axes.scatter(Y[i, 0], Y[i, 1], color='green')
    # plt.show()
    fig.savefig(title+'.png')


def data_visualization(cluster_data,cluster_label,title):
    fig = plt.figure()
    axes = fig.add_subplot(111)
    for i in range(len(cluster_data)):
        if cluster_label[i] == 0:
            axes.scatter(cluster_data[i,0],cluster_data[i,1],color='red')
        if cluster_label[i] == 1:
            axes.scatter(cluster_data[i,0],cluster_data[i,1],color='green')
    fig.savefig(title+'.png')


# 原始数据降维后（三维）可视化
def data_visualization_3D(file, labelfile):
    class_labels = get_class_label(labelfile)
    cluster_data = get_data(file)
    X = np.array(cluster_data)
    tsne = manifold.TSNE(n_components=3,init='pca',random_state=0)
    Y = tsne.fit_transform(X)
    fig = plt.figure(figsize=(8,8))
    axes = fig.add_subplot(211,projection='3d')
    for i in range(len(cluster_data)):
        if class_labels[i] == 0:
            axes.scatter(Y[i, 0], Y[i, 1], Y[i, 2], color='red')
        if class_labels[i] == 1:
            axes.scatter(Y[i, 0], Y[i, 1], Y[i, 2], color='green')
    axes.view_init(4, -72)  # 初始化视角
    plt.show()


# 生成聚类结果的2D图
def cluster_visualization_2D(clusterdata,clusterlabels,title):
    """
    :param clusterdata:数据点
    :param clusterlabels: 聚类结果标签
    :return:
    """
    X = np.array(clusterdata)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    Y = tsne.fit_transform(X)
    fig = plt.figure()
    axes = fig.add_subplot(111)
    for i in range(len(clusterdata)):
        if clusterlabels[i] == 0:
            axes.scatter(Y[i, 0], Y[i, 1], color='red')
        if clusterlabels[i] == 1:
            axes.scatter(Y[i, 0], Y[i, 1], color='green')
    plt.savefig(title+'.png')
    plt.show()
    # plt.savefig(str(int(time.time())) + 'cluster_visualization_2D.png')


# 生成聚类结果的3D图
def cluster_visualization_3D(clusterdata,clusterlabels):
    X = np.array(clusterdata)
    tsne = manifold.TSNE(n_components=3, init='pca', random_state=0)
    Y = tsne.fit_transform(X)
    fig = plt.figure(figsize=(8, 8))
    axes = fig.add_subplot(211, projection='3d')
    for i in range(len(clusterdata)):
        if clusterlabels[i] == 0:
            axes.scatter(Y[i, 0], Y[i, 1], Y[i, 2], color='red')
        if clusterlabels[i] == 1:
            axes.scatter(Y[i, 0], Y[i, 1], Y[i, 2], color='green')
    axes.view_init(4, -72)  # 初始化视角
    plt.show()


# 随机生成times次聚类结果，簇数随机选择
def C_Kmeans(cluster_data,times):
    result = []
    while times:
        # k = random.randint(2, 50)
        k = 2
        labels = generate_cluster(cluster_data, k)
        result.append(labels)
        times -= 1
    return result


# 计算准确率,仅适用于二分类
def Accuracy_binary(class_labels,result):
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


# 计算准确率,仅适用于二分类
def Accuracy_binary_1(class_labels,result):
    index_ = list(set(class_labels))
    result_index = list(set(result))
    N = len(class_labels)
    class_label = np.array(class_labels)
    label = np.array(result)
    # print(np.where(class_label==1)[0].tolist())
    class_0 = set(np.where(class_label==index_[0])[0].tolist())
    class_1 = set(np.where(class_label==index_[1])[0].tolist())
    if len(result_index) == 1:
        result_0 = set(np.where(label==result_index[0])[0].tolist())
        num1 = len(class_0.intersection(result_0))
        num2 = len(class_1.intersection(result_0))
        acc = max(num1,num2)/N
        return acc
    # print('class_0:',class_0)
    # print('class_1:',class_1)
    result_0 = set(np.where(label==result_index[0])[0].tolist())
    # print('result0',result_0)
    result_1 = set(np.where(label==result_index[1])[0].tolist())
    # print('result_1',result_1)

    num1 = len(class_0.intersection(result_0))+len(class_1.intersection(result_1))
    num2 = len(class_0.intersection(result_1)) + len(class_1.intersection(result_0))
    acc = max(num1,num2)/N
    # l1 = set(np.where(class_label==1)[0].tolist()).intersection(np.where(label==1)[0].tolist())
    # l2 = set(np.where(class_label==0)[0].tolist()).intersection(np.where(label==0)[0].tolist())
    # l3 = set(np.where(class_label==0)[0].tolist()).intersection(np.where(label==1)[0].tolist())
    # l4 = set(np.where(class_label==1)[0].tolist()).intersection(np.where(label==0)[0].tolist())
    # num = max(len(l1.union(l2)),len(l3.union(l4)))
    # acc = float(num)/N
    return acc


def Accuracy_multi_class(class_labels,result):
    label_set = set(class_labels)
    label_array = np.array(class_labels)
    result_set = set(result)
    result_array = np.array(result)
    contrast_class_list = []
    for cell in label_set:
        r = np.where(label_array==cell)
        contrast_class_list.append(set(r[0]))
    contrast_result_list=[]
    for cell in result_set:
        r = np.where(result_array==cell)
        contrast_result_list.append(set(r[0]))
    print(contrast_class_list)
    print(contrast_result_list)
    max_num = 0.
    exist_list = []
    for la in contrast_class_list:
        num = 0
        x = {}
        for re in contrast_result_list:
            if len(la.intersection(re)) > num and re not in exist_list:
                num = len(la.intersection(re))
                x = re
        exist_list.append(x)
        max_num += num
    acc = max_num/len(class_labels)
    return acc





# 获取共识聚类结果的标签
def Clustering_result(M):
    # 因为结果是二分类，所以创建两个集合，存储样本的序号
    l1 = []
    for i in range(len(M)):
        l1.append(M[:,i])
    X = np.array(l1)
    kmeans = KMeans(n_clusters=2).fit(X)
    return list(kmeans.labels_)


# 生成M和C集合，M为must-link约束，C为cannot-link约束
def generate_M_C_old(file, constrains_nums):
    data = pd.read_csv(file,delimiter=',',encoding='utf-8',header=0)
    # print(list(data['Death']))
    labels = list(data['Death'])
    n = len(labels)
    # 随机生成20个随机数
    M = []
    C = []
    # nums = set(np.random.randint(0, n, constrains_nums))
    import random
    nums = random.sample(range(0, n), constrains_nums)
    # print(nums)
    for i in nums:
        for j in nums:
            if i!=j:
                if labels[i]==labels[j]:
                    M.append([i,j])
                else:
                    C.append([i,j])
    return M,C


# 生成M和C集合
def generate_M_C_new(file,condition_num):
    '''
    :param file:标签文件
    :param condition_num:设置的条件的数量
    :return:
    '''
    data = pd.read_csv(file,delimiter=',',encoding='utf-8',header=0)
    labels = list(data['Death'])
    # 随机生成20个随机数
    M = []
    C = []
    n = len(labels)
    while condition_num:
        randnum = random.sample(range(0,n),2)
        randnum.sort()
        # print(randnum)
        if randnum not in M and randnum not in C:
            if labels[randnum[0]] == labels[randnum[1]]:
                M.append(randnum)
            else:
                C.append(randnum)
            condition_num -= 1

    return M,C
# # 定义函数Normalized Mutual Information(NMI)作为实验结果评测值
# def NMI(labels_A,labels_B):
#     n = len(labels_A)
#     Ck = set(labels_A)
#     Cm = set(labels_B)
#     # NMI计算中分子的值
#     sum_factor0 = 0.
#     for k in Ck:
#         for m in Cm:
#             indexs_k = getindex(k,labels_A)
#             indexs_m = getindex(m,labels_B)
#             nk = len(indexs_k)
#             nm = len(indexs_m)
#             n_km = len(indexs_k.intersection(indexs_m))
#             factor0 = n*n_km/(nk*nm)
#             element = math.log(factor0)
#             sum_factor0 += element
#     # 分母中第一个元素
#     denominator_0 = 0.
#     for k in Ck:
#         indexs_k = getindex(k,labels_A)
#         nk = len(indexs_k)
#         denominator_0 += nk*math.log(nk/n)
#     # 分母中第二个元素
#     denominator_1 = 0.
#     for m in Cm:
#         indexs_m = getindex(m,labels_B)
#         nm = len(indexs_m)
#         denominator_1 += nm*math.log(nm/n)
#
#     result = sum_factor0/math.sqrt(denominator_0*denominator_1)
#     return result


def getindex(k,labels):
    result = set()
    n = len(labels)
    for i in range(n):
        if labels[i] == k:
            result.add(i)
    return result


# 根据txt文件，生成权重变化趋势图
def show_weight_trends(file,title):
    data = np.loadtxt(file, delimiter=',')
    n = data.shape[1]
    m = data.shape[0]
    x = list(range(0, n))
    plt.style.use('fivethirtyeight')
    fig, ax = plt.subplots()
    plt.ylim(-0.1,1)
    for i in range(m):
        ax.plot(x, data[i, :], label=str(i), linewidth=1)
    # plt.legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(title)
    plt.savefig(title+'.png')


def generate_CCM_list(partitions):
    '''
    construct m cluster aggregation matrix Mi
    :param part: 即partition
    :return: 返回嵌套列表形式的矩阵
    '''
    M_list = []
    for part in partitions:
        n = len(part)
        # print(n)
        M = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if part[i] == part[j]:
                    M[i][j] = 1.0
                else:
                    M[i][j] = 0.00000001
        M_list.append(M)
    return M_list

if __name__ == '__main__':
    x = np.array([1,2,3])
    y = np.array([4,5,6])
    r = expDistance1(x,y)
    print(r)
    r2 = expDistance(x,y)
    print(r2)
    r3 = KLdivergence_new(x,y)
    print(r3)
    r4 = Euclidean_distance(x,y)
    print(r4)
    r5 = np.linalg.norm(x-y)
    print(r5)

    # M = [[1,1,0,0],[1,1,0,0],[0,0,1,1],[0,0,1,1]]
    # result = Fieldervector(M)
    # Eigenvector_image(result,'111')
    exit()
    # f1 = '../data/GBM/Gene.csv'
    # f2 = '../data/GBM/Survival.csv'
    # # data_visualization_3D(f1,f2)
    # data = array(get_data(f1))
    # # print(array(data).shape)
    # print('data_shape',data.shape)
    # cc = np.take(data,indices=[1,2],axis=0)
    # print(cc[0,:])
    # print(cc.shape[0])
    # for cell in data:
    #     print(cell)
    # print(data)
    # l1 = [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
    # l2 = [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1]
    l1 = [1,1,1,1]
    l2 = [1,1,0,0]
    r = Accuracy_binary_1(l1,l2)
    r1 = Accuracy_binary(l1,l2)
    print(r)
    print(r1)
    exit()
    # # print(r)
    # from sklearn.metrics import normalized_mutual_info_score as nmi
    # r = nmi(l1,l2)
    # r1 = nmi(l2,l1)
    # print(r)
    # print(r1)
    # filename0 = '../data/Kidney/Methy.csv'
    # filename1 = '../data/Kidney/Mirna.csv'
    # filename2 = '../data/Kidney/Gene.csv'
    filename0 = '../data/Lung/Methy.csv'
    filename1 = '../data/Lung/Mirna.csv'
    filename2 = '../data/Lung/Gene.csv'
    data0 = array(get_data(filename0))
    data1 = array(get_data(filename1))
    data2 = array(get_data(filename2))

    print(data0.shape)
    print(data1.shape)
    print(data2.shape)
