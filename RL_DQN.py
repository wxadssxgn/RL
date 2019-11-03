#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
import matplotlib.pyplot as plt

# Define object and initial geometry
obj = np.array([2.65, 1.95]).reshape(1, 2)
init = np.array([3.35, 2.75]).reshape(1, 2)

# Obtain the Absolute Value of S Parameter
def Get_S_Para(s_parameter):
    s_abs = np.zeros([s_parameter.shape[0], s_parameter.shape[1], 2])
    for i in range(s_parameter.shape[0]):
        temp = np.zeros([s_parameter.shape[1], 2])
        for j in range(s_parameter.shape[1]):
            temp[j][0] = abs(complex(s_parameter[i][:, 1][j], s_parameter[i][:, 2][j]))
            temp[j][1] = abs(complex(s_parameter[i][:, 3][j], s_parameter[i][:, 4][j]))
        s_abs[i] = temp
    return s_abs

# Get the geometry and spectrum Data
spectrum = np.load(file='HYB_s.npy')
spectrum = Get_S_Para(spectrum)
geometry = np.load(file='HYB_g.npy')[:, 0: 2]
geometry = geometry[:, np.newaxis, :]

def get_index(s):
    for i in range(len(geometry)):
        if (geometry[i] == np.array(s).astype(float).round(2).reshape(1, 2)).all():
            return i

obj_index = get_index(obj)
init_index = get_index(init)
obj_spec = spectrum[obj_index]
init_spec = spectrum[init_index]

# Define Hyperparameters
n_states = 2
actions = ['R+0.05', 'R-0.05', 'r+0.05', 'r-0.05']
epsilon = 0.9
alpha = 0.1
gamma = 0.9
memory_capacity = 100
LR = 0.01
batch_size = 20

class model(nn.Module):
    def __init__(self, ):
        super(model, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(2, 20), nn.Dropout(0), nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Linear(20, 20), nn.Dropout(0), nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.Linear(20, 20), nn.Dropout(0), nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(nn.Linear(20, 4))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

class dqn():
    def __init__(self, ):
        self.eval_model = model()
        self.target_model = model()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.target_replace_iteration = 10
        self.memory = np.zeros((memory_capacity, 1, n_states * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_model.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, s):
        if np.random.uniform() < epsilon:
            action_values = self.eval_model.forward(s)
            a_index = torch.max(action_values, 1)[1].numpy()
            return a_index
        else:
            a_index = np.random.randint(0, actions)
            return a_index
    
    def env_feedback(self, s, a):
        s_ = s.clone()
        if a == 0:
            s_[0, 0] = s_[0, 0] + 0.05
        elif a == 1:
            s_[0, 0] = s_[0, 0] - 0.05
        elif a == 2:
            s_[0, 1] = s_[0, 1] + 0.05
        else:
            s_[0, 1] = s_[0, 1] - 0.05
        if s_[0, 1] >= s_[0, 0] or s_[0, 0] > 3.80 or s_[0, 1] < 1.00:
            s_ = s.clone()
        return s_

    def store_transition(self, s, a, r, s_):
        s = np.array(s).astype(float).round(2).reshape(1, 2)
        s_ = np.array(s_).astype(float).round(2).reshape(1, 2)
        a = np.array(a).astype(int).reshape(1, 1)
        r = np.array(r).astype(float).reshape(1, 1)
        transition = np.hstack((s, a, r, s_))
        index = self.memory_counter % memory_capacity
        self.memory[index] = transition
        self.memory_counter += 1

    def reward(self, s, s_):
        s_index = get_index(s)
        s_index_ = get_index(s_)
        s_spec = spectrum[s_index]
        s_spec_ = spectrum[s_index_]
        s_spec = np.concatenate((s_spec[:, 0], s_spec[:, 1]), axis=0)
        s_spec_ = np.concatenate((s_spec_[:, 0], s_spec_[:, 1]), axis=0)
        s_error = ((np.concatenate((obj_spec[:, 0], obj_spec[:, 1]), axis=0) - s_spec) ** 2).sum() / len(obj_spec)
        s_error_ = ((np.concatenate((obj_spec[:, 0], obj_spec[:, 1]), axis=0) - s_spec_) ** 2).sum() / len(obj_spec)
        r1 = min(-np.log(s_error), 100)
        r2 = min(-np.log(s_error_), 100)
        r = r2 - r1
        return r

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.target_replace_iteration == 0:
            self.target_model.load_state_dict(self.eval_model.state_dict())
        self.learn_step_counter += 1
        # sample batch transitions
        sample_index = np.random.choice(memory_capacity, batch_size)
        b_memory = self.memory[sample_index]
        b_s = torch.FloatTensor(b_memory[:, :, :n_states])
        b_a = torch.LongTensor(b_memory[:, :, n_states:n_states+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, :, n_states+1:n_states+2])
        b_s_ = torch.FloatTensor(b_memory[:, :, -n_states:])
        # q_eval w.r.t the action in experience
        q_eval = self.eval_model(b_s).gather(2, b_a).reshape(batch_size, 1)
        q_next = self.target_model(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r.reshape(batch_size, 1) + gamma * q_next.max(2)[0].view(batch_size, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        print('loss:', loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

dqn_model = dqn()

s_tmp = np.array(init)
s_tmp = torch.Tensor(s_tmp)
for i in range(5000):
    a_tmp = dqn_model.choose_action(s_tmp)
    s_tmp_ = dqn_model.env_feedback(s_tmp, a_tmp)
    r_tmp = dqn_model.reward(s_tmp, s_tmp_)
    dqn_model.store_transition(s_tmp, a_tmp, r_tmp, s_tmp_)
    s_tmp = s_tmp_.clone()
    if dqn_model.memory_counter > memory_capacity:
        dqn_model.learn()

print('end')
