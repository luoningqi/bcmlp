#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 18:34:38 2018

@author: luoningqi
"""
import math
import numpy as np
import copy
from funcs import activity
from funcs import absAdd
from funcs import absCut
from funcs import drop
from funcs import getMatF
from funcs import getRandomW
from sensitivity import bcmlp_sensitivity_the

'''
        cal sensitivity of
    1. fully-connected mlps
    2. locally-connected mlps
    3. dropout
    4. BCMLP (simulation)
    5. BCMLP (theoretical)
        when input
    N number of nodes per layer
    err input deviation
    w weights
'''
def simulation(N,err,w,xget):
    # init param
    L = math.floor(math.log(N,2))
    # init input
    X1 = np.zeros((L+1,N+1))
    X2 = np.zeros((L+1,N+1))
    X1[0] = np.random.uniform(0,1,size=N+1)
    X2[0] = X1[0]+err
    nw = drop(N,L,copy.deepcopy(w))
    # cal sensitivity
    ans1 = absCut(fcmlp_sensitivity(N,L,w,copy.deepcopy(X1)),fcmlp_sensitivity(N,L,w,copy.deepcopy(X2)),N)
    ans2 = absCut(lcmlp_sensitivity(N,L,w,copy.deepcopy(X1)),lcmlp_sensitivity(N,L,w,copy.deepcopy(X2)),N)
    ans3 = absCut(fcmlp_sensitivity(N,L,nw,copy.deepcopy(X1)),fcmlp_sensitivity(N,L,nw,copy.deepcopy(X2)),N)
    ans4 = absCut(bcmlp_sensitivity_sim(N,L,w,copy.deepcopy(X1)),bcmlp_sensitivity_sim(N,L,w,copy.deepcopy(X2)),N)
    ans5 = absAdd(bcmlp_sensitivity_the(N,L,w,copy.deepcopy(X2)-copy.deepcopy(X1),xget),N)
    return ans1,ans2,ans3,ans4,ans5

''' BCMLP's sensitivity (simulation) '''
def bcmlp_sensitivity_sim(N,L,w,X):
    for l in range(1,L+1):
        for i in range(1,N+1):
            j = i+((-1)**math.floor((i-1)/(2**(l-1))))*(2**(l-1))
            X[l][i] = activity(w[l][i][i]*X[l-1][i]+w[l][i][j]*X[l-1][j])
    return X[L]

''' locally connected multilater perceptrons '''
def lcmlp_sensitivity(N,L,w,X):
    for l in range(1,L+1):
        for i in range(1,N+1):
            if i==1:
                j = 2
                X[l][i] = activity(w[l][i][i]*X[l-1][i]+w[l][i][j]*X[l-1][j])
            elif i==N:
                j = N-1
                X[l][i] = activity(w[l][i][i]*X[l-1][i]+w[l][i][j]*X[l-1][j])
            else:
                X[l][i] = activity(w[l][i][i]*X[l-1][i]+w[l][i][i-1]*X[l-1][i-1]+w[l][i][i+1]*X[l-1][i+1])
    return X[L]

''' fully connected layer '''
def fcl_sensitivity(N,L,w,X):
    for i in range(1,N+1):
        for j in range(1,N+1):
            X[L] += w[i][j]*X[0][j]
        X[L][i] = activity(X[L][i])
    return X[L]

''' fully connected multilater perceptrons '''
def fcmlp_sensitivity(N,L,w,X):
    for l in range(1,L+1):
        for i in range(1,N+1):
            for j in range(1,N+1):
                X[l][i] += w[l][i][j]*X[l-1][j]
            X[l][i] = activity(X[l][i])
    return X[L]

if __name__=="__main__":
    steps = 10000
    xget = getMatF()
    names = ['fcmlp_sensitivity',
             'lcmlp_sensitivity',
             'drop_sensitivity',
             'bcmlp_sensitivity_sim',
             'bcmlp_sensitivity_the']
    for N in [4,8,16,32]:
        for err in [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01]:
            print('-------------------')
            print('N: ',N)
            print("err: ",err)
            # init param
            L = int(math.log(N,2))
            ans = np.zeros((5))
            # cal
            for i in range(steps):
                w = getRandomW(L,N)
                temp = simulation(N,err,w,xget)
                ans += temp
            # show
            ans /= steps
            for j in range(5):
                print(names[j])
                print(ans[j])