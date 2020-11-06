# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 10:22:32 2020

@author: Administrator
"""

import numpy as np
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score,auc,roc_curve
import scipy.spatial.distance as dis
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt  

def coranking_matrix(D_high, D_low):
    N = D_high.shape[0]
    high_rank = D_high.argsort(axis=1).argsort(axis=1)
    low_rank = D_low.argsort(axis=1).argsort(axis=1)

    Q, _, _ = np.histogram2d(
        high_rank.flatten(), low_rank.flatten(), bins=N)
#    Q = Q[1:, 1:]
    
    Q_X=np.zeros((1,N))
    for K in range(1,N):
        Q_X[0][K]=sum(sum(Q[:K,:K]))/K/N
    
    Q_NX=list(Q_X[0][1:])
    Q_NX.append(1)
    return Q, Q_NX

def normalized_stress(D_high, D_low):
    return np.sum((D_high - D_low)**2) / np.sum(D_high**2)

def trustworthiness(D_high,D_low,k):
    N = D_high.shape[0]
    nn_orig = D_high.argsort()
    nn_proj = D_low.argsort()
    knn_orig = nn_orig[:, :k + 1][:, 1:]
    knn_proj = nn_proj[:, :k + 1][:, 1:]
    sum_i = 0
    for i in range(N):
        U = np.setdiff1d(knn_proj[i], knn_orig[i])
        sum_j = 0
        for j in range(U.shape[0]):
            sum_j += np.where(nn_orig[i] == U[j])[0] - k
        sum_i += sum_j
    return float((1 - (2 / (N * k * (2 * N - 3 * k - 1)) * sum_i)).squeeze())

def continuity(D_high,D_low,k):
    N = D_high.shape[0]

    nn_orig = D_high.argsort()
    nn_proj = D_low.argsort()

    knn_orig = nn_orig[:, :k + 1][:, 1:]
    knn_proj = nn_proj[:, :k + 1][:, 1:]

    sum_i = 0

    for i in range(N):
        V = np.setdiff1d(knn_orig[i], knn_proj[i])

        sum_j = 0
        for j in range(V.shape[0]):
            sum_j += np.where(nn_proj[i] == V[j])[0] - k

        sum_i += sum_j

    return float((1 - (2 / (N * k * (2 * N - 3 * k - 1)) * sum_i)).squeeze())

def shepard_diagram_correlation(D_high,D_low):
    return stats.spearmanr(D_high, D_low)[0]

def neighborhood_hit(D_low, y, k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(D_low, y)

    Dis,neighbors = knn.kneighbors(D_low, return_distance=True)
    M_NH = np.mean(np.mean((y[neighbors] == np.tile(y.reshape((-1, 1)), k)).astype('uint8'), axis=1))
    return M_NH, Dis

def D_f1(y_train,y_test,X_test_label,k):
    Y_new = dis.squareform(dis.pdist(np.vstack((y_test, y_train))))
    test_len = y_test.shape[0]
    y_new_test = Y_new[test_len:, 0:test_len]
    y_new_test = y_new_test.T
    y_new_test.sort(axis=1)
    D_test = y_new_test[:, :k].sum(axis=1)
    y_new_train = Y_new[test_len:, test_len:]
    y_new_train.sort(axis=1)
    D_train = y_new_train[:, :k].sum(axis=1)
#    D_limit = np.mean(D_train) + 3*np.std(D_train)
    D_limit = compute_threshold(D_train,bw=None,alpha=0.95)
    y_test_pred = (D_test > D_limit).astype(int)
    y_test_label = (X_test_label > 0).astype(int)
    
    Q_f1 = f1_score(y_test_label, y_test_pred, pos_label=1)
    fpr, tpr, thresholds = roc_curve(y_test_label, D_test, pos_label=1)
    Q_auc = auc(fpr, tpr)
    return D_train,D_test,D_limit,fpr,tpr,Q_f1, Q_auc

def compute_threshold(data,bw=None,alpha=0.95):
    ##先用sklearn的KDE拟合
    ##首先将数据尺度缩放到近似无穷大，然后根据近似微分求解
    data=data.reshape(-1,1)
    Min=np.min(data)
    Max=np.max(data)
    Range=Max-Min
    ##起点和重点
    x_start=Min-Range
    x_end=Max+Range
    ###nums越大之后估计的累积概率越大
    nums=2**12
    dx=(x_end-x_start)/(nums-1)
    data_plot=np.linspace(x_start,x_end,nums)
    if bw is None:
        ##最佳带宽选择
        ##参考：Adrian W, Bowman Adelchi Azzalini
        # - Applied Smoothing Techniques for Data Analysis_
        # The Kernel Approach with S-Plus Illustrations (1997)
        ##章节2.4.2 Normal optimal smoothing,中位数估计方差效果更好，
        #与matlab的ksdensity一致
        data_median=np.median(data)
        new_median=np.median(np.abs(data-data_median))/0.6745
        ##np.std(data,ddof=1)当ddof=1时计算无偏标准差，即除以n-1，为0时除以n
        bw=new_median*((4/(3*data.shape[0]))**0.2)
    # print(bw)
    # print(data.shape)
    ##
    kde=KernelDensity(kernel='gaussian',bandwidth=bw).fit(data.reshape(-1,1))
    ###得到的是log后的概率密度，之后要用exp恢复
    log_pdf = kde.score_samples(data_plot.reshape(-1, 1))
    pdf=np.exp(log_pdf)
#    print(pdf.shape[0])
    ##画概率密度图
#    plt.plot(data_plot,pdf)
#    plt.show()
    ##CDF：累积概率密度
    CDF=0
    index=0
    try:
        while CDF<=alpha:
            CDF+=pdf[index]*dx
            index+=1
    except IndexError:
        index=pdf.shape[0]-1
#    print(index)
    return data_plot[index]