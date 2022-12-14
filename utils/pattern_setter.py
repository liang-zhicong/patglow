import os
import torch
import numpy as np
import math

def get_pattern(patterns, arr):               # input : (?, 1, 9) / output : (?, 10) 
    l = len(arr)
    #在56个pattern set中找到次数最多的k个pattern
    for j in range(l):
        found_flag = 0
        for i in range(len(patterns)):
            if np.array_equal([patterns[i][0:9]], arr[j].tolist()):
                patterns[i][9] = patterns[i][9]+1
                found_flag = 1
                break;

        if(found_flag == 0):
            y = np.c_[arr[j], [1]]
            patterns.append(y.tolist()[0])
    return patterns    


def top_4(arr):                     # input : (d, ch, 1, 9) / output : (d*ch, 1, 9)
    arr = arr.reshape(-1,1,9)#拉直，d ch 拉直成d*ch
    arr = abs(arr)
    for i in range(len(arr)):
        arr[i][0][4] = 0    #中间核设为0
        x = arr[i].copy()
        x.sort()
        arr[i]=np.where(arr[i]<x[0][6], 0, 1)#排序后把最大的3个核设为1
        arr[i][0][4] = 1    #中间核设为1
    return arr                     


def pattern_setter(model, num_sets=8):
    patterns = [[0,0,0,0,0,0,0,0,0,0]]
    
    for name, param in model.named_parameters():
        if name.split('.')[-1] == "weight" and name.split('.')[-2] !='mid_conv' and len(param.shape) == 4 and param.shape[2] == 3 and 'downsample' not in name:#中间核1*1不剪
        # if name.split('.')[-1] == "weight"  and len(param.shape) == 4 and param.shape[2] == 3 and 'downsample' not in name:
            #限制：weight层，param是的维数是4，第二个维度是3，3*3核卷积,中間曾不剪
            print(f'name:{name}')
            par=param.detach().numpy()
            patterns=get_pattern(patterns, top_4(par))
 
    patterns = np.array(patterns, dtype='int')
    patterns = patterns[patterns[:,9].argsort(kind='mergesort')]
    patterns = np.flipud(patterns)

    pattern_set = patterns[:num_sets,:9]
    # print(pattern_set)
    
    return pattern_set

# new !!!!!!!!!  I wanna test it!
def top_4_pat(arr, pattern_set):    # input arr : (d, ch, 3, 3) or (d, ch, 1, 1)   pattern_set : (6~8, 9)
    if arr.shape[2] == 3:
        cpy_arr = arr.copy().reshape(-1, 9)
        new_arr = np.zeros(cpy_arr.shape)
        pat_set = pattern_set.copy().reshape(-1, 9)
        #拉直操作
        for i in range(len(cpy_arr)):
            pat_arr = cpy_arr[i] * pat_set #1*9的向量 * 8*9的向量
            pat_arr = np.linalg.norm(pat_arr, axis=1)#计算范数
            pat_idx = np.argmax(pat_arr)#选择范数最大的矩阵

            new_arr[i] = cpy_arr[i] * pat_set[pat_idx]

        new_arr = new_arr.reshape(arr.shape)
        return new_arr
    else:
        return arr
        

def top_k_kernel(arr, perc):    # input (d, ch, 3, 3)
    if arr.shape[2] == 1:
        new_arr = arr.copy().reshape(-1, 1) #拉直成(d*ch,1)的矩阵   # (d*ch, 1)
    elif arr.shape[2] == 3:
        new_arr = arr.copy().reshape(-1, 9)    # (d*ch, 9)
    else:
        return arr

    k = math.ceil(arr.shape[0] * arr.shape[1] / perc)#prec 剪除通道的閾值，1是全保留
    l2_arr = np.linalg.norm(new_arr, axis=1)#计算l2范数
    threshold = l2_arr[np.argsort(-l2_arr)[k-1]]#对-L2_arr进行排序得到矩阵序号
    l2_arr = l2_arr >= threshold#取出l2_arr中从大到小的k-1位，每一位（每一维向量）代表一个矩阵，因此这是关于连通性的剪枝，
    
    if arr.shape[2] == 1:
        new_arr = new_arr.reshape(-1) * l2_arr

    elif arr.shape[2] == 3:
        l2_arr = l2_arr, l2_arr, l2_arr, l2_arr, l2_arr, l2_arr, l2_arr, l2_arr, l2_arr
        l2_arr = np.transpose(np.array(l2_arr))#转置
        new_arr = new_arr * l2_arr#矩阵相乘，相当于取l2_arr的值，其他的剪掉
   
    new_arr = new_arr.reshape(arr.shape)
    return new_arr


##### for 'main_swp.py' #####
def top_4_pat_swp(arr, pattern_set):   # input arr : (d, ch, 3, 3) or (d, ch, 1, 1)  pattern_set : (6~8, 9)
    if arr.shape[2] == 3:
        cpy_arr = arr.copy().reshape(len(arr), -1, 9)
        new_arr = np.zeros(cpy_arr.shape)
        pat_set = pattern_set.copy().reshape(-1, 9)
        pat_rst = np.zeros(len(pat_set))

        pat_arr = 0
        for i in range(len(cpy_arr)):
            for j in range(len(pat_set)):
                pat_arr = cpy_arr[i] * pat_set[j]
                pat_rst[j] = np.linalg.norm(pat_arr.reshape(-1))
        
            pat_idx = np.argmax(pat_rst)
            new_arr[i] = cpy_arr[i] * pat_set[pat_idx]

        new_arr = new_arr.reshape(arr.shape)
        return new_arr
    else:
        return arr



""" my mistake1... should use tensor / torch calculation! (for speed)

def top_4_pat(arr, pattern_set):    # input arr : (d, ch, 3, 3)   pattern_set : (6~8, 9) (9 is 3x3)
    cpy_arr = arr.copy().reshape(-1, 1, 9)
    new_arr = np.zeros(cpy_arr.shape)

    for i in range(len(cpy_arr)):
        max = -1
        for j in range(len(pattern_set)):
            pat_arr = cpy_arr[i] * pattern_set[j]
            pat_l2 = np.linalg.norm(cpy_arr[i])
            
            if pat_l2 > max:
                max = pat_l2
                new_arr[i] = pat_arr
        
    new_arr = new_arr.reshape(arr.shape)
    return new_arr


def top_k_kernel(arr, perc):    # input (d, ch, 3, 3)
    k = math.ceil(arr.shape[0] * arr.shape[1] / perc)
    new_arr = arr.copy().reshape(-1, 1, 9)    # (d*ch, 1, 9)
    l2_arr = np.zeros(len(new_arr))

    for i in range(len(new_arr)):
        l2_arr[i] = np.linalg.norm(new_arr[i]) 
        
    threshold = l2_arr[np.argsort(-l2_arr)[k-1]]    # top k-th l2-norm

    for i in range(len(new_arr)):
        new_arr[i] = new_arr[i] * (l2_arr[i] >= threshold)
    
    new_arr = new_arr.reshape(arr.shape)
    return new_arr
"""





