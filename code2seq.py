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

class pycolor:
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    PURPLE = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    END = '\033[0m'
    BOLD = '\038[1m'
    UNDERLINE = '\033[4m'
    INVISIBLE = '\033[08m'
    REVERCE = '\033[07m'

def Keywords():
    return [
        'auto',
        'break',
        'case',
        'char',
        'const',
        'continue',
        'default',
        'double',
        'do',
        'else',
        'enum',
        'extern',
        'float',
        'for',
        'goto',
        'if',
        'int',
        'long',
        'register',
        'return',
        'short',
        'signed',
        'sizeof',
        'static',
        'struct',
        'switch',
        'typedef',
        'union',
        'unsigned',
        'void',
        'volatile',
        'while']

def Characters():
    return [
        ' ',
        '!',
        '?',
        '_',
        '\"',
        '#',
        '$',
        '%',
        '&',
        '\'',
        '(',
        ')',
        '*',
        '+',
        ',',
        '-',
        '.',
        '/',
        '0',
        '1',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7',
        '8',
        '9',
        ':',
        ';',
        '<',
        '=',
        '>',
        '@',
        'A',
        'B',
        'C',
        'D',
        'E',
        'F',
        'G',
        'H',
        'I',
        'J',
        'K',
        'L',
        'M',
        'N',
        'O',
        'P',
        'Q',
        'R',
        'S',
        'T',
        'U',
        'V',
        'W',
        'X',
        'Y',
        'Z',
        '[',
        '\\',
        ']',
        '^',
        '`',
        'a',
        'b',
        'c',
        'd',
        'e',
        'f',
        'g',
        'h',
        'i',
        'j',
        'k',
        'l',
        'm',
        'n',
        'o',
        'p',
        'q',
        'r',
        's',
        't',
        'u',
        'v',
        'w',
        'x',
        'y',
        'z',
        '{',
        '|',
        '}',
        '~']

def Words():
    return Keywords()+Characters()

def Print(code,underlines=[]):
    line_n = 1
    line = ''
    size = int(math.log10(code.count('\n')))+1
    
    for w in code:
        if w == '\n' : 
            n = str(line_n)
            line = ' '*(size-len(n))+n+' '+line
            if line_n in underlines:
                line = pycolor.RED+line+pycolor.END
            print line
            line = ''
            line_n += 1
        else: line+=w
    print

def del_c_comments(code):
    return re.sub('/\*[\s\S]*?\*/|//.*','',code)

def del_space(code):
    flag = True
    res = []
    for i in range(len(code)):
        if code[i]=='"' and flag==True: flag = False
        elif code[i]=='"' and flag==False: flag = True
        if code[i]==' ' and flag==True: continue
        if code[i]=='\n' or code[i]=='\t':continue
        res.append(code[i])
    return res

def get_functions(code,keywords):
    func = re.findall('\w+[ \t\n]*\(',code)
    functions = []
    for f in func:
        f = re.sub('[ \t\n]*\(','',f)
        if f not in functions and f not in keywords:
            functions.append(f)
    return functions

def is_variable(s):
    if len(s)==0: return False
    if s[0].isdigit():return False
    return True

def is_letter(w):
    if len(w)!=1:return False
    if w=='_' or w.isalnum():return True
    return False

# return list of tuple of func_name and var_name
def get_variables(code,functions,keywords,data_types = ['char','int','short','long','float','double']):
    variables = []
    tmp = []
    func = ''
    bracket = 0
    i=0
    while i<len(code)-1:
        c = code[i]
        if c=='{':bracket+=1
        if c=='}':bracket-=1
        if bracket==0:func=""
        if c in functions and bracket==0:
            func = c
            j = i+1
            parenthesis = 0
            fragment = ''
            while code[j]==' ' or code[j]=='\t' or code[j]=='\n':j+=1
            while True:
                if code[j]=='(': parenthesis+=1
                if code[j]==')': parenthesis-=1
                fragment+=code[j];
                j+=1
                if parenthesis==0: break
            while code[j]==' ' or code[j]=='\t' or code[j]=='\n':j+=1
            i=j
            var = re.findall('[a-zA-Z0-9_]*',fragment)
            for v in var:
                if v is not "" and v not in keywords and is_variable(v) and not v in variables:
                    variables.append((v,func))
        elif c in data_types:
            j=i+1
            fragment = ''
            while code[j]==' ' or code[j]=='*' or code[j]=='\n' or code[j]=='\t':j+=1
            k=j+1
            while code[k]==' ' or code[k]=='\n' or code[k]=='\t':k+=1
            if (code[j] in functions and code[k]=='(') or (code[j] in keywords):
                i = j
                continue
            j=i
            while True:
                j+=1
                if code[j]=='\n' or code[j]=='\t' or code[j]==' ':continue
                fragment+=code[j]
                if code[j]==';':break
            fragment = re.findall('[a-zA-Z0-9_]+\=|[a-zA-Z0-9_]+\,|[a-zA-Z0-9_]+\;|[a-zA-Z0-9_]+\[',fragment)
            for f in fragment:
                v = f[:len(f)-1]
                if is_variable(v) and (v,func) not in variables and (v,'') not in variables:
                    variables.append((v,func))
            i=j+1
        else:
            i+=1
    return variables

def refactoring(code):
    i = 0
    defines = []
    while i<len(code):
        w = code[i]
        if w == '#':
            j=i+1
            while code[j]==' ' or code[j]=='\t':j+=1
            d = ''
            while code[j]!=' ' and code[j]!='\t':
                d+=code[j]
                j+=1
            if d=='define':
                while code[j]==' ' or code[j]=='\t':j+=1
                fragment = ''
                while code[j]!='\n':
                    fragment+=code[j]
                    j+=1
                fragment = re.split(' |\t',fragment)
                defines.append((fragment[0],' '.join(fragment[1:])))
        i+=1
    i = 0
    target = 'typedef'
    while i<len(code)-len(target):
        w = code[i:i+len(target)]
        if w==target:
            j=i+len(target)
            while code[j]==' ' or code[j]=='\t':j+=1
            fragment = ''
            while code[j]!=';':
                fragment+=code[j]
                j+=1
            fragment = re.split(' |\t',fragment)
            size = len(fragment)
            defines.append((fragment[size-1],' '.join(fragment[:size-1])))
        i+=1
    res = re.sub('\#[ \t]*define.*','',code)
    res = re.sub('typedef.*','',res)
    for a,b in defines:
        res = re.sub(a,b,res)
    return res

def c2seq(code):
    code = del_c_comments(code)
    code = refactoring(code)
    keywords = Keywords()
    functions = get_functions(code,keywords)
    functions_keywords = keywords + functions
    functions_keywords.sort()
    functions_keywords.reverse()
    if len(functions)>10: return None,None,None
    seq = []
    semicolon = False
    i = 0
    while i<len(code):
        flag = True
        if code[i]=='"':
            if semicolon:semicolon = False
            else : semicolon = True
        for w in functions_keywords:
            if semicolon:break
            if w == '':continue
            if len(code)<i+len(w):continue
            if w == ''.join(code[i:i+len(w)]) and not is_letter(code[i+len(w)]) and not is_letter(code[max(i-1,0)]):
                seq.append(w)
                i+=len(w)
                flag = False
                break
        if flag:
            seq.append(code[i])
            i+=1
    variables = get_variables(seq,functions,keywords)
    if len(variables)>20: return None,None,None
    var = variables[:]
    var.sort()
    var.reverse()
    seq = del_space(seq)
    res = []
    i = 0
    semicolon = False
    while i<len(seq):
        flag = True
        if seq[i]=='"':
            if semicolon:semicolon = False
            else : semicolon = True
        for w,f in var:
            if i==0:break
            if semicolon:break
            if w == '':continue
            if len(seq)<i+len(w):continue
            if w == ''.join(seq[i:i+len(w)]) and not is_letter(seq[i+len(w)]) and not is_letter(seq[i-1]):
                res.append(w)
                i+=len(w)
                flag = False
                break
        if flag:
            res.append(seq[i])
            i+=1
    return res,functions,variables

def numbering(f):
    seq,functions,variables = c2seq(f)
    words = Words()
    s = np.ndarray((len(seq),), dtype=np.int32)
    if seq==None and functions==None and variables==None: return s
    func = ''
    count = 0
    semicolon = False
    for i in range(len(seq)):
        if seq[i]=='{':count+=1
        if seq[i]=='}':count-=1
        if seq[i]=='"':
            if semicolon:semicolon = False
            else : semicolon = True
        if seq[i] in functions and not semicolon:
            index = functions.index(seq[i])
            if count==0:func = functions[index]
            s[i]=index
        elif (seq[i],func) in variables and not semicolon:
            index = variables.index((seq[i],func))
            s[i]=index+10
        elif (seq[i],'') in variables and not semicolon:
            index = variables.index((seq[i],''))
            s[i]=index+10
        else :
            s[i]=words.index(seq[i])+30
    return s

def main():
    input_file = open('test.c','r')
    f = input_file.read()
    input_file.close()
    Print(f)
    seq,f,v = c2seq(f)
    print seq
    print f
    print v

if __name__ == '__main__':
    main()

