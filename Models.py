# -*- coding: utf-8 -*-

import io
import re
import math
import sys
import time
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import chainer
from chainer import cuda,Function,gradient_check, \
                    Variable,optimizers,serializers,utils
from chainer import Link,Chain,ChainList
import chainer.functions as F
import chainer.links as L

'''
# Bidirectional LSTM for logistic Regression
class BLSTM(Chain):
    def __init__(self, vocab_size, hidden_size, ratio = 0.0, train = True):
        super(BLSTM, self).__init__(
            embed = L.EmbedID(vocab_size, hidden_size),
            f = L.LSTM(hidden_size, hidden_size),
            b = L.LSTM(hidden_size, hidden_size),
            linear = L.Linear(hidden_size*2, 1)
        )
        if train:
            for param in self.params():
                param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.ratio = ratio
        self.train = train
    
    def __call__(self, x):
        self.reset_state()
        y_f = []
        y_b = []
        for i in range(x.size-1):
            _x = self.embed(Variable(np.array(([x[i]]),dtype = np.int32)))
            y_f.append(self.f(F.dropout(_x,ratio = self.ratio)))
        for i in range(x.size-1,0,-1):
            _x = self.embed(Variable(np.array(([x[i]]),dtype = np.int32)))
            y_b.append(self.b(F.dropout(_x,ratio = self.ratio)))

        loss = None
        for i in range(len(y_f)):
            for j in range(len(y_b)-1-i,-1,-1):
                h1 = y_f[i]
                h2 = y_b[j]
                h = F.concat([h1,h2])
                y = self.linear(F.dropout(h,ratio = self.ratio))
                t = Variable(np.array(([len(x)-i-j-2]),dtype = np.int32))
                if loss is not None:
                    loss += F.mean_square_error(y,t)
                else :
                    loss = F.mean_square_error(y,t)
        return loss

    def cal(self,f_words,b_words):
        h1 = None
        for i in range(f_words):
            _x = self.embed(Variable(np.array(([f_words[i]]),dtype = np.int32)))
            h1 = self.f(F.dropout(_x,ratio = self.ratio))
        h2 = None
        for i in range(b_words):
            _x = self.embed(Variable(np.array(([b_words[i]]),dtype = np.int32)))
            h2 = self.f(F.dropout(_x,ratio = self.ratio))
        h = F.concat([h1,h2])
        y = self.linear(F.dropout(h,ratio = self.ratio))
        return y

    def perplexity(self,x):
        sum_log_perp = 0
        loss = self.__call__(x)
        sum_log_perp += loss.data
        return math.exp(float(sum_log_perp) / (x.size - 2.0))

    def reset_state(self):
        self.f.reset_state()
        self.b.reset_state()
'''
'''
# Bidirectional LSTM with Attention for Language Modeling
class BLSTM(Chain):
    def __init__(self, vocab_size, hidden_size, ratio = 0.5, train = True):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        super(BLSTM, self).__init__(
            embed = L.EmbedID(vocab_size, hidden_size),
            f = L.LSTM(hidden_size,hidden_size),
            b = L.LSTM(hidden_size,hidden_size),
            f_W1 = L.Linear(hidden_size,hidden_size),
            b_W1 = L.Linear(hidden_size,hidden_size),
            f_W2 = L.Linear(hidden_size,hidden_size),
            b_W2 = L.Linear(hidden_size,hidden_size),
            #f_W = L.Linear(hidden_size,vocab_size),
            #b_W = L.Linear(hidden_size,vocab_size),
            linear = L.Linear(hidden_size + hidden_size, vocab_size)
        )
        if train:
            for param in self.params():
                param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
        self.ratio = ratio
        self.train = train
        self.f_gh = []
        self.b_gh = []

    def __call__(self, x):
        self.reset_state()
        y_f = []
        y_b = []
        for i in range(len(x)-2):
            _x = self.embed(Variable(np.array(([x[i]]),dtype = np.int32)))
            h1 = self.f(F.dropout(_x,ratio = self.ratio))
            h2 = F.tanh(self.f_W1(self.f_ct(h1.data[0]))+self.f_W2(h1))
            #_y = self.f_W(F.dropout(h2,ratio = self.ratio))
            y_f.append(h2)
            self.f_gh.append(np.copy(h1.data[0]))
        for i in range(len(x)-1,1,-1):
            _x = self.embed(Variable(np.array(([x[i]]),dtype = np.int32)))
            h1 = self.b(F.dropout(_x,ratio = self.ratio))
            h2 = F.tanh(self.b_W1(self.b_ct(h1.data[0]))+self.b_W2(h1))
            #_y = self.b_W(F.dropout(h2,ratio = self.ratio))
            y_b.append(h2)
            self.b_gh.append(np.copy(h1.data[0]))

        y = [self.linear(F.dropout(F.concat([yf,yb]),ratio = self.ratio)) for yf,yb in zip(y_f,y_b[::-1])]
        if not self.train: return y
        loss = None
        for i in range(1,len(x)-1):
            t = Variable(np.array(([x[i]]),dtype = np.int32))
            if loss is not None:
                loss += F.softmax_cross_entropy(y[i-1],t)
            else : 
                loss = F.softmax_cross_entropy(y[i-1],t)
        return loss

    def parse(self,x):
        _y = self.__call__(x)
        res = [F.softmax(y) for y in _y]
        return res

    def perplexity(self,x):
        sum_log_perp = 0
        loss = self.__call__(x)
        sum_log_perp += loss.data
        return math.exp(float(sum_log_perp) / (x.size - 2.0))

    def reset_state(self):
        self.f.reset_state()
        self.b.reset_state()
        self.f_gh = []
        self.b_gh = []

    def f_ct(self,ht):
        s = 0.0
        ct = np.zeros(self.hidden_size)
        for i in range(len(self.f_gh)):
            s += np.exp(ht.dot(self.f_gh[i]))
        for i in range(len(self.f_gh)):
            alp = np.exp(ht.dot(self.f_gh[i]))/s
            ct += alp*self.f_gh[i]
        return Variable(np.array([ct]).astype(np.float32))
    
    def b_ct(self,ht):
        s = 0.0
        ct = np.zeros(self.hidden_size)
        for i in range(len(self.b_gh)):
            s += np.exp(ht.dot(self.b_gh[i]))
        for i in range(len(self.b_gh)):
            alp = np.exp(ht.dot(self.b_gh[i]))/s
            ct += alp*self.b_gh[i]
        return Variable(np.array([ct]).astype(np.float32))
'''


# Bidirectional LSTM for Language Modeling
class BLSTM(Chain):
    def __init__(self, n_vocab, n_units, train = True):
        self.n_units = n_units
        self.n_vocab = n_vocab
        super(BLSTM, self).__init__(
            embed = L.EmbedID(n_vocab, n_units),
            f = L.LSTM(n_units, n_units),
            b = L.LSTM(n_units, n_units),
            linear = L.Linear(n_units + n_units, n_vocab)
        )
        if train:
            for param in self.params():
                param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
            self.ratio = 0.5
        else: self.ratio = 0.0
        self.train = train
    
    def __call__(self, x):
        self.reset_state()
        y_f = []
        y_b = []
        for i in range(len(x)-2):
            _x = self.embed(Variable(np.array(([x[i]]),dtype = np.int32)))
            y_f.append(self.f(F.dropout(_x,ratio = self.ratio)))
        for i in range(len(x)-1,1,-1):
            _x = self.embed(Variable(np.array(([x[i]]),dtype = np.int32)))
            y_b.append(self.b(F.dropout(_x,ratio = self.ratio)))

        y = [self.linear(F.dropout(F.concat([yf,yb]),ratio = self.ratio)) for yf,yb in zip(y_f,y_b[::-1])]
        if not self.train: return y
        loss = None
        for i in range(1,len(x)-1):
            t = Variable(np.array(([x[i]]),dtype = np.int32))
            if loss is not None:
                loss += F.softmax_cross_entropy(y[i-1],t)
            else : 
                loss = F.softmax_cross_entropy(y[i-1],t)
        return loss

    def parse(self,x):
        _y = self.__call__(x)
        res = [F.softmax(y) for y in _y]
        return res

    def perplexity(self,x):
        sum_log_perp = 0
        loss = self.__call__(x)
        sum_log_perp += loss.data
        return math.exp(float(sum_log_perp) / (x.size - 2.0))

    def reset_state(self):
        self.f.reset_state()
        self.b.reset_state()

'''
# LSTM with Attention type alpha for Language Modeling
class LSTM(Chain):
    def __init__(self, vocab_size, hidden_size, train=True):
        super(LSTM, self).__init__(
            Embed = L.EmbedID(vocab_size,hidden_size),
            H = L.LSTM(hidden_size,hidden_size),
            W1 = L.Linear(hidden_size,hidden_size),
            W2 = L.Linear(hidden_size,hidden_size),
            W = L.Linear(hidden_size,vocab_size)
        )
        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
        if train: self.ratio = 0.5
        else : self.ratio = 0.0
        self.train = train
        self.gh = []
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def set_dropout_ratio(self, ratio):
        self.ratio = ratio

    def get_dropout_ratio(self):
        return self.ratio

    def reset_state(self):
        self.H.reset_state()
        self.gh = []

    def __call__(self, x):
        e = self.Embed(x)
        h1 = self.H(F.dropout(e,ratio = self.ratio))
        h2 = F.tanh(self.W1(self.ct(h1.data[0]))+self.W2(h1))
        y = self.W(F.dropout(h2,ratio = self.ratio))
        self.gh.append(np.copy(h1.data[0]))
        if self.train: return y
        return F.softmax(y)

    def ct(self,ht):
        s = 0.0
        ct = np.zeros(self.hidden_size)
        for i in range(len(self.gh)):
            s += np.exp(ht.dot(self.gh[i]))
        for i in range(len(self.gh)):
            alp = np.exp(ht.dot(self.gh[i]))/s
            ct += alp*self.gh[i]
        return Variable(np.array([ct]).astype(np.float32))

    def print_gh(self):
        print self.gh[0] 

    def perplexity(self,s):
        self.reset_state()
        self.set_dropout_ratio(0.0)
        sum_log_perp = 0
        for i in range(s.size - 1):
            x = chainer.Variable(np.asarray(s[i:i + 1]))
            t = chainer.Variable(np.asarray(s[i + 1:i + 2]))
            loss = F.softmax_cross_entropy(self.__call__(x),t)
            sum_log_perp += loss.data
        self.set_dropout_ratio(0.5)
        return math.exp( float(sum_log_perp) / (s.size - 1))


# LSTM with Attention type beta for Language Modeling
class LSTM(Chain):
    def __init__(self, vocab_size, hidden_size, train=True):
        super(LSTM, self).__init__(
            Embed = L.EmbedID(vocab_size,hidden_size),
            H = L.LSTM(hidden_size,hidden_size), 
            W = L.Linear(hidden_size,vocab_size),
            H1 = L.Linear(hidden_size,hidden_size),
            H2 = L.Linear(hidden_size,hidden_size),
            att = L.Linear(hidden_size,1)
        )
        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
        if train: self.ratio = 0.0
        else : self.ratio = 0.0
        self.train = train
        self.gh = []
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def set_dropout_ratio(self, ratio):
        self.ratio = ratio

    def get_dropout_ratio(self):
        return self.ratio

    def reset_state(self):
        self.H.reset_state()
        self.gh = []

    def __call__(self, x):
        e = self.Embed(x)
        h1 = self.H(F.dropout(e,ratio = self.ratio))
        ws = []
        sum_w = 0
        sum_w = Variable(np.zeros((1,1), dtype='float32'))
        for i in range(len(self.gh)):
            w = F.exp(self.att(F.tanh(self.H1(self.gh[i])+self.H2(h1))))
            sum_w += w
            ws.append(w)
        h2 = Variable(np.zeros((1,self.hidden_size), dtype='float32'))
        for w,h in zip(ws,self.gh):
            h2 += F.reshape(F.batch_matmul(h,(w/sum_w)),(1,self.hidden_size))
        if len(self.gh)==0:h2 = h1
        y = self.W(F.dropout(h2,ratio = self.ratio))
        self.gh.append(h1)
        if self.train: return y
        return F.softmax(y)

    def print_gh(self):
        print self.gh[0]
    
    def perplexity(self,s):
        self.reset_state()
        self.set_dropout_ratio(0.0)
        sum_log_perp = 0
        for i in range(s.size - 1):
            x = chainer.Variable(np.asarray(s[i:i + 1]))
            t = chainer.Variable(np.asarray(s[i + 1:i + 2]))
            loss = F.softmax_cross_entropy(self.__call__(x),t)
            sum_log_perp += loss.data
        self.set_dropout_ratio(0.5)
        return math.exp( float(sum_log_perp) / (s.size - 1))
'''

# Simple LSTM for Language Modeling
class LSTM(Chain):
    def __init__(self, n_vocab, n_units, train=True):
        super(LSTM, self).__init__(
            embed = L.EmbedID(n_vocab,n_units),
            l1 = L.LSTM(n_units,n_units),
            l2 = L.LSTM(n_units,n_units),
            l3 = L.Linear(n_units,n_vocab)
        )
        for param in self.params():
            param.data[...] = np.random.uniform(-0.1, 0.1, param.data.shape)
        if train: self.ratio = 0.5
        else : self.ratio = 0.0
        self.train = train

    def set_dropout_ratio(self, ratio):
        self.ratio = ratio

    def get_dropout_ratio(self):
        return self.ratio

    def __call__(self, x):
        h0 = self.embed(x)
        h1 = self.l1(F.dropout(h0,ratio = self.ratio))
        h2 = self.l2(F.dropout(h1,ratio = self.ratio))
        y = self.l3(F.dropout(h2,ratio = self.ratio))
        if self.train: return y
        return F.softmax(y)

    def reset_state(self):
        self.l1.reset_state()
        self.l2.reset_state()

    def perplexity(self,s):
        self.reset_state()
        self.set_dropout_ratio(0.0)
        sum_log_perp = 0
        for i in range(s.size - 1):
            x = chainer.Variable(np.asarray(s[i:i + 1]))
            t = chainer.Variable(np.asarray(s[i + 1:i + 2]))
            loss = F.softmax_cross_entropy(self.__call__(x),t)
            sum_log_perp += loss.data
        self.set_dropout_ratio(0.5)
        return math.exp( float(sum_log_perp) / (s.size - 1))
