#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 18:12:14 2018

@author: luoningqi
"""

'''
BCMLP
'''

import tensorflow as tf
import numpy as np

class bcmlp:
    def __init__(self, L):
        N = 2**L
        lr = 0.500
        self.inp, self.lab = self.getplaceholder(N)
        self.arch = self.architecture(L,N)
        self.outp = self.getOutput(L,N)
        self.loss, self.train = self.trainOp(lr, 'l1')
        
    def getplaceholder(self, N):
        inp = tf.placeholder(tf.float32, shape=(N))
        lab = tf.placeholder(tf.float32, shape=())
        return inp, lab
    
    def architecture(self, L,N):
        # init param
        data = []
        with tf.variable_scope('Line0'):
            dataline = [0]
            for i in range(1,N+1):
                dataline.append(self.inp[i-1])
            data.append(dataline)
        # connect nodes
        for l in range(1,L+1):
            dataline = [0]
            with tf.variable_scope('Line'+str(l)):
                for i in range(1,N+1):
                    with tf.variable_scope('Node'+str(i)):
                        j = int(i+(-1)**np.floor((i-1)/2**(l-1))*2**(l-1))
                        zf = (-1)**(int((i-1)/2**(l-1))%2)
                        kernelii = tf.Variable(1.0*zf,name='weights'+str(i)+str(i))
                        kernelij = tf.Variable(-1.0*zf,name='weights'+str(i)+str(j))
                        biases = tf.Variable(0.0,name='biases'+str(i))
                        dataline.append(tf.nn.sigmoid(data[l-1][i]*kernelii+data[l-1][j]*kernelij+biases))
            data.append(dataline)
        return data
    
    def getOutput(self, L,N):
        # output self.arch[L][1:] -> 1
        with tf.variable_scope('output'):
            kernel = tf.Variable(tf.truncated_normal([N],dtype=tf.float32),name='weights')
            biases = tf.Variable(0.0,name='biases')
            output = tf.nn.sigmoid(tf.reduce_sum(self.arch[L][1:]*kernel)+biases)
        return output
    
    def trainOp(self, lr, losstype):
        if losstype=='l1':
            loss = tf.abs(self.outp-self.lab)
        else:
            loss = (self.outp-self.lab)**2
        train = tf.train.GradientDescentOptimizer(lr).minimize(loss)
        return loss,train
    