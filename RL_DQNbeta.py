# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
import matplotlib.pyplot as plt

obj = np.array([2.65, 1.95]).reshape(1, 2)
init = np.array([3.00, 2.20]).reshape(1, 2)

def get_s_para(s_parameter):
    s_abs = np.zeros([s_parameter.shape[0], s_parameter.shape[1], 2])
    for i in range(s_parameter.shape[0]):
        temp = np.zeros([s_parameter.shape[1], 2])
        for j in range(s_parameter.shape[1]):
            temp[j][0] = abs(complex(s_parameter[i][:, 1][j], s_parameter[i][:, 2][j]))
            temp[j][1] = abs(complex(s_parameter[i][:, 3][j], s_parameter[i][:, 4][j]))
        s_abs[i] = temp
    return s_abs

spectrum = np.load(file='HYB_s.npy')
geometry = np.load(file='HYB_g.npy')[:, 0: 2]
spectrum = get_s_para(spectrum)

n_states = 2
actions = ['R+0.05', 'R-0.05', 'r+0.05', 'r-0.05']
memory = np.zeros((len(spectrum)*4 , 1, n_states*2 + 2))

for j in range(len(geometry)):
    geo_tmp = geometry[j]
    i = 4 * j
    memory[i+0, 0,  0: 2] = geometry[j]
    memory[i+1, 0,  0: 2] = geometry[j]
    memory[i+2, 0,  0: 2] = geometry[j]
    memory[i+3, 0,  0: 2] = geometry[j]
    memory[i+0, 0,  2   ] = 0
    memory[i+1, 0,  2   ] = 1
    memory[i+2, 0,  2   ] = 2
    memory[i+3, 0,  2   ] = 3
    memory[i+0, 0, -2:  ] = np.array([geometry[j, 0]+0.05, geometry[j, 1]])
    memory[i+1, 0, -2:  ] = np.array([geometry[j, 0]-0.05, geometry[j, 1]])
    memory[i+2, 0, -2:  ] = np.array([geometry[j, 0], geometry[j, 1]+0.05])
    memory[i+3, 0, -2:  ] = np.array([geometry[j, 0], geometry[j, 1]-0.05])

memory = memory.round(2)

hyb_memory = memory[0]

for i in range(1, len(memory)):
    meo_tmp = memory[i]
    if meo_tmp[0, 0] > meo_tmp[0, 1] and meo_tmp[0, -2] > meo_tmp[0, -1] \
        and meo_tmp[0, -2] <= 3.80 and meo_tmp[0, -1] >= 1.00:
        hyb_memory = np.append(hyb_memory, meo_tmp, axis=0)

hyb_memory = np.expand_dims(hyb_memory, 1)

def get_index(s):
    for i in range(len(geometry)):
        if (geometry[i] == np.array(s).astype(float).round(2).reshape(1, 2)).all():
            return i

obj_index = get_index(obj)
init_index = get_index(init)
obj_spec = spectrum[obj_index]
init_spec = spectrum[init_index]

def reward(s, s_):
    '''
    s : current state
    s_: next state
    '''
    s_index = get_index(s)
    s_index_ = get_index(s_)
    if s_index != s_index_:
        s_spec  = spectrum[s_index]
        s_spec_ = spectrum[s_index_]
        s_spec  = np.concatenate((s_spec[:, 0], s_spec[:, 1]), axis=0)
        s_spec_ = np.concatenate((s_spec_[:, 0], s_spec_[:, 1]), axis=0)
        # s_error = ((np.concatenate((obj_spec[:, 0], obj_spec[:, 1]), axis=0) - s_spec) ** 2).sum() / len(obj_spec)
        # s_error_ = ((np.concatenate((obj_spec[:, 0], obj_spec[:, 1]), axis=0) - s_spec_) ** 2).sum() / len(obj_spec)
        # r1 = -np.log(max(s_error, 1e-4))
        # r2 = -np.log(max(s_error_, 1e-4))
        # r = (r2 - r1) * 15
        # r = -np.log(max(s_error_, 1e-4)) - 3
        s_error_ = ((np.concatenate((obj_spec[:, 0], obj_spec[:, 1]), axis=0) - s_spec_)).sum() / len(obj_spec)
        r = s_error_
        return r
    else:
        r = -10
        return r

for i in range(len(hyb_memory)):
    r_tmp = reward(hyb_memory[i][0][0: 2], hyb_memory[i][0][-2: ])
    hyb_memory[i][0][3] = r_tmp
    # print(i)

for j in range(len(hyb_memory)):
    print(hyb_memory[j])

# np.savetxt('HYB_memory.txt', hyb_memory.squeeze(1), fmt='%.2f')
# np.save('HYB_memory.npy', hyb_memory)


print('end')

