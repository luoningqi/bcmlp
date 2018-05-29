#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 19:11:28 2018

@author: luoningqi
"""
from bcmlp import bcmlp
import numpy as np
import tensorflow as tf
from funcs import XOR4
from funcs import writeAns

def getData(N,err):
    inl = np.zeros((N))
    for i in range(N):
        inl[i] = np.random.randint(0,2)
    tag = getTag(inl)
    for i in range(N):# add noise
        itag = np.random.randint(0,3)
        if itag==0:
            inl[i] += err
        elif itag==1:
            inl[i] -= err
        else:
            inl[i] += 0.0
    return inl,tag

def getTag(inl):#XOR4
    return XOR4(inl[0],inl[1],inl[2],inl[3])

def test(sess, model, N, err):
    # init param
    rou = 100
    deviation = 0.0
    # test
    for i in range(rou):
        p,q = getData(N, err)
        outp = sess.run(model.outp, {model.inp: p, model.lab: q})
        deviation += abs(round(outp)-q)
    # return accuracy
    return 1-deviation/rou

if __name__=="__main__":
    # init state
    model = bcmlp(2)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    ''' debug
    tf.summary.scalar('loss', model.loss)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./tfgraphs', sess.graph)
    '''
    err = 0.20 #[0.20,0.30,0.40]:
    ans = []
    for i in range(10000):
        # get test data
        p,q = getData(4,err)
        ''' train '''
        sess.run(model.train, {model.inp: p, model.lab: q})
        ''' test '''
        testacc = test(sess, model, 4, err)
        print('steps:'+str(i))
        print(testacc)
        ans.append(testacc)
    # record
    writeAns(ans,'bcmlp_noisytask_err'+str(int(err*100)))