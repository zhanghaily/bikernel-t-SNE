# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 09:26:43 2020

@author: Administrator
"""
import numpy as np
import scipy.spatial.distance as dis
import pandas as pd
import Q_metrics
import DataIn
import matplotlib.pyplot as plt
import matplotlib as mpl

filepath = 'F:\\E0507886\\python work\\Paper 4\\result_data'
#dataset_name = ['mnist_1','mnist_2','norb_1','cnae9_1','har_1','sms_1','bank_1']
##dataset_name = ['mnist_0','mnist','norb_0','cnae9_0','har_0','sms_0','bank_0']
#methods = ['bikernel','kernel','lion','DL','UMAP','AE','PCA','KPCA']
#m=0
#F = np.zeros((56,2))
#out_train = np.zeros((40,5))
#out_test = np.zeros((40,5))
#for i in range(7):#np.arange(2,6,1):
#    if i==0:
#        X_train,X_train_label,X_test,X_test_label = DataIn.mnist(1)
##        X_test=X_test[:5000,:]
##        X_test_label = X_test_label[:5000]
#        X_test_label[X_test_label!=1]=2
#        X_test_label[X_test_label==1]=0
#        X_test_label[X_test_label==2]=1
#    if i==1:
#        X_train,X_train_label,X_test,X_test_label = DataIn.mnist(2)
#        X_test_label[X_test_label<=3]=0
#        X_test_label[X_test_label>2]=1
#
#    if i==2:
#        X_train,X_train_label,X_test,X_test_label = DataIn.norb(1)
#        X_test_label[X_test_label!=5]=0
#        X_test_label[X_test_label==5]=1
#    if i==3:
#        X_train,X_train_label,X_test,X_test_label = DataIn.cnae9(1)
#        X_test_label[X_test_label!=2]=0
#        X_test_label[X_test_label==2]=1
#    if i==4:
#        X_train,X_train_label,X_test,X_test_label = DataIn.har(1)
#        X_test_label[X_test_label!=6]=0
#        X_test_label[X_test_label==6]=1
#    if i==5:
#        X_train,X_train_label,X_test,X_test_label = DataIn.sms(1)
#    if i==6:
#        X_train,X_train_label,X_test,X_test_label = DataIn.bank(1)
#    label = X_train_label.copy()
##%%
#    for j in range(8):
#        f_train = filepath+'\\'+methods[j]+'_'+dataset_name[i]+'_train.txt'
#        y_train = np.loadtxt(f_train)
#        f_test = filepath+'\\'+methods[j]+'_'+dataset_name[i]+'_test.txt'
#        y_test = np.loadtxt(f_test)
##        y_test = y_test[:5000,:]
#     #%       
#        k=10
#        #~~~~~~~~~~test~~~~~~~~~~
##        D_high_test = dis.squareform(dis.pdist(X_test))
##        D_low_test = dis.squareform(dis.pdist(y_test))
##        Q, Q_NX = Q_metrics.coranking_matrix(D_high_test, D_low_test)
###        M_sigma = Q_metrics.normalized_stress(D_high_test, D_low_test)
##        M_c = Q_metrics.continuity(D_high_test, D_low_test, k)
##        M_t = Q_metrics.trustworthiness(D_high_test,D_low_test,k)
##        M_S = Q_metrics.shepard_diagram_correlation(D_high_test.reshape(-1,1),D_low_test.reshape(-1,1))
##        M_NH,Dis_test = Q_metrics.neighborhood_hit(D_low_test, X_test_label, k)
##        q_test = np.array(Q_NX)
#        d_train,d_test,d_limit,fpr,tpr,F1,AUC=Q_metrics.D_f1(y_train,y_test,X_test_label,k)
#        F[m,:]=[F1,AUC]
##        
##        out_test[m,:] = [M_t, M_c,  M_NH, q_test[k],M_S] 
##        #~~~~~~~~~~~~train~~~~~~~~~~~
##        D_high_train = dis.squareform(dis.pdist(X_train))
##        D_low_train = dis.squareform(dis.pdist(y_train))
##        Q, Q_NX = Q_metrics.coranking_matrix(D_high_train, D_low_train)
###        M_sigma = Q_metrics.normalized_stress(D_high_train, D_low_train)
##        M_c = Q_metrics.continuity(D_high_train, D_low_train, k)
##        M_t = Q_metrics.trustworthiness(D_high_train,D_low_train,k)
##        M_S = Q_metrics.shepard_diagram_correlation(D_high_train.reshape(-1,1),D_low_train.reshape(-1,1))
##        M_NH,Dis_train = Q_metrics.neighborhood_hit(D_low_train, X_train_label, k)
##        q_train = np.array(Q_NX)
##        out_train[m,:] = [M_t, M_c, M_NH, q_train[k],M_S]
#        m=m+1

#Q=np.loadtxt(filepath+'\\Q_test.txt')
#fig,ax = plt.subplots(5,1, figsize=(4, 9))
#M = ['','Bi-kernel t-SNE','Kernel t-SNE','Lion t-SNE','DL t-SNE','UMAP','AE','PCA','KPCA']
#dataset_name = ['','MNIST','NORB','CNAE9','HAR']
#for i in range(5):
#    A=Q[:,i].reshape(-1,8)
#    ax[i].matshow(A,cmap='Blues')
#    if i ==0:
#        ax[i].set_xticklabels(M,fontsize=5)
#    else:
#        ax[i].set_xticklabels([],fontsize=5)
#    ax[i].set_yticklabels(dataset_name,fontsize=5)
#    for x in range(len(A)):
#        for y in range(A.shape[1]):
#            ax[i].annotate(A[x,y],xy=(y,x),horizontalalignment='center',verticalalignment='center',fontsize=5)
#plt.subplots_adjust(bottom=0.1,right=0.85,wspace=0,hspace=0.1)
#%%
Q_test = np.loadtxt(filepath+'\\Q_test.txt')
Q_train = np.loadtxt(filepath+'\\Q_train.txt')
fig,ax = plt.subplots(4,2, figsize=(5.8, 7.9))
M = ['','Bikernel\nt-SNE','Kernel\ntSNE','LION\ntSNE','DL\ntSNE','UMAP','AE','PCA','KPCA']
dataset_name = ['MNIST_0','NORB_0','CNAE9_0','HAR_0']
Q_name = ['','T','C','NH','QN','S']
a_test=Q_test.reshape((4,8,5))
a_train=Q_train.reshape((4,8,5))
norm = mpl.colors.Normalize(vmin=0, vmax=1)
for i in range(4):
    A_train=a_train[i,:,:].T
    A_test=a_test[i,:,:].T
    cmap=mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#FfFFFf','#d1ffbd','#bdf8a3', '#48D1CC'], 256)
    ax[i][0].matshow(A_train,cmap=cmap,norm=norm)
    ax[i][1].matshow(A_test,cmap=cmap,norm=norm)
#    ax[i][0].set_axis_off()
#    ax[i][1].set_axis_off()
    if i ==0:
        ax[i][0].set_xticklabels(M,rotation=0,fontsize=6)
        ax[i][0].tick_params(axis='both', which='both', labelsize=6,
               bottom=False, top=True, labeltop=True,
               left=True, right=False, labelleft=True)
    else:
        ax[i][0].set_xticklabels([],fontsize=6)
        ax[i][0].tick_params(axis='both', which='both', labelsize=6,
               bottom=False, top=False, labeltop=True,
               left=True, right=False, labelleft=True)
    ax[i][0].set_yticklabels(Q_name,fontsize=6)
    ax[i][0].set_ylabel(dataset_name[i],fontsize=7)
    for x in range(len(A_train)):
        for y in range(A_train.shape[1]):
            ax[i][0].annotate(A_train[x,y],xy=(y,x),horizontalalignment='center',verticalalignment='center',fontsize=6)
    
    if i ==0:
        ax[i][1].set_xticklabels(M,rotation=0,fontsize=6)
        ax[i][1].tick_params(axis='both', which='both', labelsize=6,
               bottom=False, top=True, labeltop=True,
               left=False, right=False, labelleft=True)
    else:
        ax[i][1].set_xticklabels([],fontsize=6)
        ax[i][1].tick_params(axis='both', which='both', labelsize=6,
               bottom=False, top=False, labeltop=True,
               left=False, right=False, labelleft=True)
    ax[i][1].set_yticklabels([],fontsize=6)
    for x in range(len(A_test)):
        for y in range(A_test.shape[1]):
            ax[i][1].annotate(A_test[x,y],xy=(y,x),horizontalalignment='center',verticalalignment='center',fontsize=6)

fig.text(0.31,0.98,'Training',ha='center',va='center',fontsize=8)
fig.text(0.77,0.98,'Testing',ha='center',va='center',fontsize=8)
plt.tick_params(axis='x', bottom=False)
plt.subplots_adjust(top=0.92,bottom=0.03,left=0.07,right=0.99,wspace=0.01,hspace=0.05)

cbar=fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax,orientation='horizontal',fraction=0.017,aspect=40,pad=0.02)
#cbar.set_ticks([0,0.3,0.6,0.9,1.2,1.5]) 
#cbar.set_ticks([0,0.2,0.4,0.6,0.8,1])
#cbar.ax.set_ylim([0,1])
cbar.ax.set_xlim([0,1])
#cbar.set_ticklabels(['0','0.2','0.4','0.6','0.8','1']) 
cbar.ax.tick_params(labelsize=7)
#%% outlier
#Q = np.loadtxt(filepath+'\\Q_outlier.txt')
#fig,ax = plt.subplots(7,1, figsize=(3, 5.7))
#M = ['','Bikernel\nt-SNE','Kernel\ntSNE','LION\ntSNE','DL\ntSNE','UMAP','AE','PCA','KPCA']
#dataset_name = ['MNIST_1','MNIST_2','NORB_1','CNAE9_1','HAR_1','SMS_1','BANK_1']
#Q_name = ['','F1','AUC']
#a=Q.reshape((7,8,2))
##a_train=Q_train.reshape((4,8,5))
#norm = mpl.colors.Normalize(vmin=0, vmax=1)
#for i in range(7):
#    A=a[i,:,:].T
##    A_test=a_test[i,:,:].T
#    ax[i].matshow(A,cmap=cmap,norm=norm)
#    if i ==0:
#        ax[i].set_xticklabels(M,rotation=0,fontsize=6)
#        ax[i].tick_params(axis='both', which='both', labelsize=6,
#               bottom=False, top=True, labeltop=True,
#               left=True, right=False, labelleft=True)
#    else:
#        ax[i].set_xticklabels([],fontsize=6)
#        ax[i].tick_params(axis='both', which='both', labelsize=6,
#               bottom=False, top=False, labeltop=True,
#               left=True, right=False, labelleft=True)
#    ax[i].set_yticklabels(Q_name,fontsize=6)
#    ax[i].set_ylabel(dataset_name[i],fontsize=7)
#    for x in range(len(A)):
#        for y in range(A.shape[1]):
#            ax[i].annotate(A[x,y],xy=(y,x),horizontalalignment='center',verticalalignment='center',fontsize=6)
#
#plt.tick_params(axis='x', bottom=False)
#plt.subplots_adjust(top=0.95,bottom=0.03,left=0.16,right=0.97,wspace=0.00,hspace=0)
##
#cbar=fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax,orientation='horizontal',fraction=0.017,aspect=40,pad=0.02)
#cbar.set_ticks([0,0.3,0.6,0.9,1.2,1.5]) 
#cbar.set_ticklabels(['0','0.2','0.4','0.6','0.8','1']) 
#cbar.ax.tick_params(labelsize=7)
#%%
#plt.savefig('F:\\E0507886\\python work\\Paper 4\\figures\\Q.tif',dpi=300)
#plt.savefig('F:\\E0507886\\python work\\Paper 4\\figures\\Q.png',dpi=300)