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
from Models import LSTM,BLSTM
from code2seq import Words,c2seq,del_c_comments,Print,refactoring,numbering

# argvs[1]:  # of units
# argvs[2]:  # of epochs
# argvs[3]:  File name

argvs = sys.argv
words = Words()
threshold = 0.05
print threshold
path = os.path.dirname(os.path.abspath(__file__))+'/models/'
vocab_size = len(words)+30
n_units = int(argvs[1])

def load_c_program():
    input_file = open(argvs[3],'r')
    code = input_file.read()
    input_file.close()
    return code

def Print_Bugs(code,bugs):
    seq,functions,variables =  c2seq(code)
    code = del_c_comments(code)
    tmp = refactoring(code)
    line_num = [0 for _ in range(len(seq))]
    j = 0
    line_n = 1
    for i in range(len(seq)):
        w_size = len(seq[i])
        while True:
            w = tmp[j:j+w_size]
            if seq[i] == w:
                line_num[i]=line_n
                break
            if tmp[j]=='\n':line_n+=1
            j+=1
    bugs_line = []
    for i in bugs:
        bugs_line.append(line_num[i])
    Print(code,bugs_line)

def BLSTM_detector(code):

    model_name = argvs[1]+'_'+argvs[2]+'.npz'
    model = BLSTM(vocab_size,n_units,False)
    if not os.path.exists(path+model_name): 
        print 'not found the model'
        return
    serializers.load_npz(path+model_name,model)

    seq,functions,variables =  c2seq(code)
    f = numbering(code)
    _y = model.parse(f)

    bugs = []
    print f
    for i in range(len(f)-2):
        y = _y[i]
        ans = y.data.argmax()
        if y.data[0][f[i+1]] < threshold:
            bugs.append(i+1)
        if f[i+1]<10: print functions[f[i+1]],y.data[0][f[i+1]],' '*4,
        elif f[i+1]<30: print variables[f[i+1]-10],y.data[0][f[i+1]],' '*4,
        else : print words[f[i+1]-30],y.data[0][f[i+1]],' '*4,
        if ans<10: print functions[ans],y.data[0][ans]
        elif ans<30: print variables[ans-10],y.data[0][ans]
        else : print words[ans-30],y.data[0][ans]
    return bugs


def LSTM_detector(code):

    model_name = argvs[1]+'_'+argvs[2]+'.npz'
    rnnlm = LSTM(vocab_size,n_units,False)
    model = L.Classifier(rnnlm)
    if not os.path.exists(path+model_name): 
        print 'not found the model'
        return
    serializers.load_npz(path+model_name,model)
    
    seq,functions,variables =  c2seq(code)
    f = numbering(code)

    bugs = []
    model.predictor.reset_state()
    for i in range(len(f)-1):
        x = Variable(np.array(([f[i]]),dtype = np.int32))
        y = model.predictor(x)
        ans = y.data.argmax()
        if y.data[0][f[i+1]]<threshold: 
            bugs.append(i+1)
            if f[i+1]<10: print functions[f[i+1]],y.data[0][f[i+1]],' '*4,
            elif f[i+1]<30: print variables[f[i+1]-10],y.data[0][f[i+1]],' '*4,
            else : print words[f[i+1]-30],y.data[0][f[i+1]],' '*4,
            if ans<10: print functions[ans],y.data[0][ans]
            elif ans<30: print variables[ans-10],y.data[0][ans]
            else : print words[ans-30],y.data[0][ans]
    return bugs


def main():
    code = load_c_program()
    bugs = LSTM_detector(code)
    #bugs = BLSTM_detector(code)
    Print_Bugs(code,bugs)

if __name__ == '__main__':
    main()

