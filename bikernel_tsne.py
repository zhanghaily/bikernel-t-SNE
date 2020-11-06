# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 19:35:56 2019
bikernel t-SNE
@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt  
import scipy.spatial.distance as dis
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import Q_metrics
import DataIn

#~~~~~~~~~~~~~import data~~~~~~~~~~
X_train,X_train_label,X_test,X_test_label = DataIn.norb(0)
#%%
train_len = X_train.shape[0]
test_len = X_test.shape[0]

### 标准化
#Mean=np.mean(X_train,axis=0)
#Zstd=np.std(X_train,axis=0)
#Zstd[Zstd==0]=1
#X_train = (X_train-Mean)/Zstd
#X_test = (X_test-Mean)/Zstd
#%%             tSNE
t_train = TSNE(n_components=2,perplexity=40,init='pca',early_exaggeration=4,learning_rate=500,method='exact' ).fit_transform(X_train)#
#plt.figure()
#plt.scatter(t_train[:,0],t_train[:,1],s=7,c=X_train_label,cmap = 'jet')
#%% ~~~~~~~CALDIS~~~~~~~~~~
X_train_new = dis.squareform(dis.pdist(X_train)) # CALDIS
t_train_new = dis.squareform(dis.pdist(t_train)) # CALDIS
#%% ~~~~~~~~PARACAL
a=X_train_new.reshape(1,-1).copy()
a.sort()
sigma = 1/np.sqrt(-2*np.log(0.6))*np.median(a[0,-10:])
a=t_train_new.reshape(1,-1).copy()
a.sort()
sigma_t = 1/np.sqrt(-2*np.log(0.6))*np.median(a[0,-10:])
#%%
K_X = np.exp(-1*X_train_new**2/(2*sigma**2))
a=K_X
K_t = np.exp(-1*t_train_new**2/(2*sigma_t**2))

A=np.dot(np.dot(np.linalg.pinv(np.dot(K_X.T,K_X)),K_X.T),K_t)
pca = PCA(n_components=2)
y_train = pca.fit_transform(K_t)

X_new = dis.squareform(dis.pdist(np.vstack((X_test,X_train))))
X_test_new = X_new[0:test_len,-train_len:]
K_new=np.exp(-1*X_test_new**2/(2*sigma**2))
K_test = np.dot (K_new, A)
y_test = pca.transform(K_test)
#%%  #~~~~~~~quality matrics calculation~~~~~~~~~~~
k=10
#~~~~~~~~~~~~train~~~~~~~~~~~
D_high_train = dis.squareform(dis.pdist(X_train))
D_low_train = dis.squareform(dis.pdist(y_train))
Q, Q_NX = Q_metrics.coranking_matrix(D_high_train, D_low_train)
M_s_train = Q_metrics.normalized_stress(D_high_train, D_low_train)
M_c_train = Q_metrics.continuity(D_high_train, D_low_train, k)
M_t_train = Q_metrics.trustworthiness(D_high_train,D_low_train,k)
M_NH_train,Dis_train = Q_metrics.neighborhood_hit(D_low_train, X_train_label, k)
q_train = np.array(Q_NX)
Q_N_train = q_train[k]

#~~~~~~~~~~test~~~~~~~~~~
D_high_test = dis.squareform(dis.pdist(X_test))
D_low_test = dis.squareform(dis.pdist(y_test))
Q, Q_NX = Q_metrics.coranking_matrix(D_high_test, D_low_test)
M_s_test = Q_metrics.normalized_stress(D_high_test, D_low_test)
M_c_test = Q_metrics.continuity(D_high_test, D_low_test, k)
M_t_test = Q_metrics.trustworthiness(D_high_test,D_low_test,k)
M_NH_test,Dis_test = Q_metrics.neighborhood_hit(D_low_test, X_test_label, k)
q_test = np.array(Q_NX)
Q_N = q_test[k]

D_train,D_test,D_limit,fpr,tpr,F1, Q_auc=Q_metrics.D_f1(y_train,y_test,X_test_label,k) #F1-score

#%%
plt.figure()
plt.scatter(y_test[:,0],y_test[:,1],s=7,c=X_test_label,cmap = 'jet')
plt.figure()
plt.scatter(y_train[:,0],y_train[:,1],s=7,c=X_train_label,cmap = 'jet')
