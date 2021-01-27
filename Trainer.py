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
from code2seq import c2seq,Words,numbering

words = Words()
argvs = sys.argv
path = os.path.dirname(os.path.abspath(__file__))
vocab_size = len(words)+30
n_units = int(argvs[1]) # argvs[1] is # of units of hidden layers
n_epoch = 30

def load_c_programs():
    print 'Loading'
    file_names = os.listdir(path+'/data/')
    files = []
    for file_name in file_names:
        print file_name
        # Exceptions
        if file_name == '.DS_Store':continue
        if file_name == '._.DS_Store':continue
        if file_name == '1209984.txt':continue
        if file_name == '1230399.txt':continue
        if file_name == '1283604.txt':continue
        if file_name == '1353076.txt':continue
        if file_name == '1367750.txt':continue
        if file_name == '1458509.txt':continue
        if file_name == '1775241.txt':continue
        if file_name == '1975251.txt':continue
        if file_name == '2113510.txt':continue
        if file_name == '2255968.txt':continue
        if file_name == '2275297.txt':continue
        if file_name == '990961.txt':continue

        input_file = open(path+'/data/'+file_name,'r')
        f = input_file.read()
        input_file.close()
        s = numbering(f)
        if s.size==0:continue
        files.append(s)
    return files

def BLSTM_training(files):

    model = BLSTM(vocab_size,n_units)
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    optimizer.compute_accuracy = False

    print 'BLSTM training starts with %d files %d epochs' % (len(files),n_epoch)
    e_size = 10
    t_size = len(files)-e_size
    start = time.time()
    training_time = 0
    print '#'*100
    for epoch in range(1,n_epoch+1):
        ppl = 0
        files[:t_size] = random.sample(files[:t_size],t_size)
        for k in range(len(files)):
            s = files[k]
            if k<t_size:
                loss = model(s)
                model.zerograds()
                loss.backward()
                loss.unchain_backward()
                optimizer.update()
                now = time.time()
                training_time += now-start
                start = now
            else :
                ppl+=model.perplexity(s)
        print 'Training time is %d min %f sec' % (training_time/60,training_time%60)
        print 'Perplexity is %.10f at epoch %d' % (ppl/(len(files)-t_size),epoch)
        model_name = argvs[1]+'_'+str(epoch)+'.npz'
        print 'Save '+model_name
        serializers.save_npz(path+'/models/'+model_name,model)
        print '#'*100

    print 'Finished'
    print 'Training time is  %d min %f sec' % (training_time/60,training_time%60)


def LSTM_training(files):

    rnnlm = LSTM(vocab_size,n_units)
    model = L.Classifier(rnnlm)
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    optimizer.compute_accuracy = False

    print 'LSTM training starts with %d files %d epochs' % (len(files),n_epoch)
    e_size = 10
    t_size = len(files)-e_size
    start = time.time()
    training_time = 0
    print '#'*100
    for epoch in range(1,n_epoch+1):
        ppl = 0
        files[:t_size] = random.sample(files[:t_size],t_size)
        for k in range(len(files)):
            s = files[k]
            rnnlm.reset_state()
            if k<t_size:
                accum_loss = 0
                for curr_id,next_id in zip(s,s[1:]):
                    x = Variable(np.array(([curr_id]),dtype = np.int32))
                    t = Variable(np.array(([next_id]),dtype = np.int32))
                    accum_loss += model(x, t)
                model.zerograds()
                accum_loss.backward()
                accum_loss.unchain_backward()
                optimizer.update()
                now = time.time()
                training_time += now-start
                start = now
            else : 
                ppl+=rnnlm.perplexity(s)
        print 'Training time is %d min %f sec' % (training_time/60,training_time%60)
        print 'Perplexity is %.10f at epoch %d' % (ppl/e_size,epoch)
        model_name = argvs[1]+'_'+str(epoch)+'.npz'
        print 'Save '+model_name
        serializers.save_npz(path+'/models/'+model_name,model)
        print '#'*100

    print 'Finished'
    print 'Training time is  %d min %f sec' % (training_time/60,training_time%60)

def main():
    files = load_c_programs()
    #LSTM_training(files)
    BLSTM_training(files)

if __name__ == '__main__':
    main()

    
