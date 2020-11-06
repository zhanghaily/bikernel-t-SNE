# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 10:26:14 2020
data import
DataIn
@author: Administrator
"""

import numpy as np

def mnist(mode):
    X_train = np.loadtxt('mnist_train_6000.txt')
    X_train_label = np.loadtxt('mnist_train_label_6000.txt')
    X_test = np.loadtxt('mnist_test_10000.txt')
    X_test_label = np.loadtxt('mnist_test_label_10000.txt')
    if mode == 1: # 1(1) vs 10(0-9)
        f = np.where(X_train_label==1)[0]
        X_train = X_train[f,:]
        X_train_label = X_train_label[f]
    if mode == 2:# 3(0,1,2) vs 10(0-9)
        f = np.r_[np.where(X_train_label==1)[0],np.where(X_train_label==0)[0],np.where(X_train_label==2)[0]]
        X_train = X_train[f,:]
        X_train_label = X_train_label[f]   
    return X_train,X_train_label,X_test,X_test_label

def bank(mode):
    data = np.loadtxt('data_banknote_authentication.txt',delimiter=',')
    X_train = data[:600,:4]
    X_train_label = data[:600,4]
    X_test = data[600:,:4]
    X_test_label = data[600:,4]
    return X_train,X_train_label,X_test,X_test_label

def norb(mode):
    X_train = np.loadtxt('norb_2000.txt')
    label = np.loadtxt('norb_label_2000.txt')
    X_train_label = label[:,3]
    X_test = np.loadtxt('norb_1000.txt')
    label = np.loadtxt('norb_label_1000.txt')
    X_test_label = label[:,3]
    if mode == 1:
        f = np.where (label[:,3] != 5)[0]
        X_train_label = label[f,3]
        X_train = X_train[f,:]
        X_test = np.loadtxt('norb_1000.txt')
        label = np.loadtxt('norb_label_1000.txt')
        X_test_label = label[:,3]
    return X_train,X_train_label,X_test,X_test_label

def cnae9(mode):
    X = np.load('cnae9_X.npy')
    Y = np.load('cnae9_Y.npy')
    X_train = X[:400,:]
    X_train_label = Y[:400]
    X_test = X[400:,:]
    X_test_label = Y[400:]
    if mode ==1:
        f=np.where(X_train_label!=2)[0]
        X_train = X_train[f,:]
        X_train_label = X_train_label[f]
    f= X_train_label.argsort()
    X_train = X_train[f,:]
    X_train_label.sort()
    f= X_test_label.argsort()
    X_test = X_test[f,:]
    X_test_label.sort()
    return X_train,X_train_label,X_test,X_test_label

def sms(mode):
    X = np.load('sms_X.npy')
    Y = np.load('sms_Y.npy')
    f = Y.argsort()
    X = X[f]
    Y.sort()
    if mode ==1:
        X_train = X[:500]
        X_train_label = Y[:500]
        X_test = X[500:,:]
        X_test_label = Y[500:]
    
    return X_train,X_train_label,X_test,X_test_label

def har(mode):
    X_train = np.loadtxt('HAR\\X_train.txt')
    X_train_label = np.loadtxt('HAR\\y_train.txt')
    X_train = X_train [:1000]
    X_train_label = X_train_label[:1000]
    X_test = np.loadtxt('HAR\\X_test.txt')
    X_test_label = np.loadtxt('HAR\\y_test.txt')
    if mode ==1:
        f=np.where(X_train_label!=6)[0]
        X_train = X_train[f,:]
        X_train_label = X_train_label[f]
    return X_train,X_train_label,X_test,X_test_label