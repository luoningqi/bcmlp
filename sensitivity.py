#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 18:21:37 2018

@author: luoningqi
"""

import math
from scipy.special import comb
from funcs import activity
from funcs import zk

'''
        calculate sensitivity (theoretical)
input: N number of nodes per layer
       L number of layers
       w weights
       X delta X (input deviation)
       xget mat for cal combinations
output:
       S sensitivity

'''

def bcmlp_sensitivity_the(N,L,w,X,xget):
    # input is X[0][i]
    # cal sensitivity
    for l in range(1,L+1):
        for i in range(1,N+1):
            j = i+((-1)**math.floor((i-1)/(2**(l-1))))*(2**(l-1))
            X[l][i]=taylor(w[l][i][i],w[l][i][j],X[l-1][i],X[l-1][j],xget,l)
    # return output
    S = X[L]
    return S

def taylor(w1,w2,cx,cy,xget,l):
    su = 0
    sss = 0
    for n in range(1,20):
        for r in range(n+1):
            cou = comb(n,r)
    
            z1 = activity(w1+w2)
            z2 = activity(w1)
            z3 = activity(w2)
            z4 = activity(0)
            if (n-2)==-1:
                sss = math.log(1+math.e**(w1+w2),math.e)-math.log(1+math.e**(w1),math.e)-math.log(1+math.e**(w2),math.e)+math.log(1+math.e**(0),math.e)            
            else:
                sss = (zk(z1,n-2,xget)-zk(z2,n-2,xget)-zk(z3,n-2,xget)+zk(z4,n-2,xget))
            temp = cou*(w1**(r-1))*(w2**(n-r-1))*sss*(cx**r)*(cy**(n-r))
            temp = temp/math.factorial(n)

            su += temp
    return su