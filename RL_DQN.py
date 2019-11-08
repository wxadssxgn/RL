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
        spectrum = np.load(file='HYB_s.npy')
        spectrum = self.get_s_para(spectrum)
        self.spectrum = spectrum
        geometry = np.load(file='HYB_g.npy')[:, 0: 2]
        self.geometry = geometry[:, np.newaxis, :]
        obj_index = self.get_index(obj)
        init_index = self.get_index(init)
        self.obj_spec = self.spectrum[obj_index]
        self.init_spec = self.spectrum[init_index]
        self.n_states = 2
        self.actions = ['R+0.05', 'R-0.05', 'r+0.05', 'r-0.05']
        self.epsilon = 0.9
        self.alpha = 0.1
        self.gamma = 0.9
        self.memory_capacity = 100
        self.LR = 0.01
        self.batch_size = 20

        self.eval_model = model()
        self.target_model = model()
        if torch.cuda.is_available():
            self.eval_model = self.eval_model.cuda()
            self.target_model = self.target_model.cuda()

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.target_replace_iteration = 10
        self.memory = np.zeros((self.memory_capacity, 1, self.n_states * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_model.parameters(), lr=self.LR)
        self.loss_func = nn.MSELoss()

    def get_index(self, s):
        for i in range(len(self.geometry)):
            if (self.geometry[i] == np.array(s).astype(float).round(2).reshape(1, 2)).all():
                return i

    def get_s_para(self, s_parameter):
        s_abs = np.zeros([s_parameter.shape[0], s_parameter.shape[1], 2])
        for i in range(s_parameter.shape[0]):
            temp = np.zeros([s_parameter.shape[1], 2])
            for j in range(s_parameter.shape[1]):
                temp[j][0] = abs(complex(s_parameter[i][:, 1][j], s_parameter[i][:, 2][j]))
                temp[j][1] = abs(complex(s_parameter[i][:, 3][j], s_parameter[i][:, 4][j]))
            s_abs[i] = temp
        return s_abs
    
    def choose_action(self, s):
        # a_index.type = array
        if np.random.uniform() < self.epsilon:
            action_values = self.eval_model.forward(s)
            a_index = torch.max(action_values, 1)[1].numpy()
            return a_index
        else:
            a_index = np.random.randint(0, 4, size=(1, ))
            return a_index
    
    def env_feedback(self, s, a):
        # s.type = Tensor
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
        index = self.memory_counter % self.memory_capacity
        self.memory[index] = transition
        self.memory_counter += 1

    def reward(self, s, s_):
        s_index = self.get_index(s)
        s_index_ = self.get_index(s_)
        s_spec = self.spectrum[s_index]
        s_spec_ = self.spectrum[s_index_]
        s_spec = np.concatenate((s_spec[:, 0], s_spec[:, 1]), axis=0)
        s_spec_ = np.concatenate((s_spec_[:, 0], s_spec_[:, 1]), axis=0)
        s_error = ((np.concatenate((self.obj_spec[:, 0], self.obj_spec[:, 1]), axis=0) - s_spec) ** 2).sum() / len(self.obj_spec)
        s_error_ = ((np.concatenate((self.obj_spec[:, 0], self.obj_spec[:, 1]), axis=0) - s_spec_) ** 2).sum() / len(self.obj_spec)
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
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        b_memory = self.memory[sample_index]
        b_s = torch.FloatTensor(b_memory[:, :, :self.n_states])
        b_a = torch.LongTensor(b_memory[:, :, self.n_states: self.n_states+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, :, self.n_states+1: self.n_states+2])
        b_s_ = torch.FloatTensor(b_memory[:, :, -self.n_states :])
        # q_eval w.r.t the action in experience
        q_eval = self.eval_model(b_s).gather(2, b_a).reshape(self.batch_size, 1)
        q_next = self.target_model(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r.reshape(self.batch_size, 1) + self.gamma * q_next.max(2)[0].view(self.batch_size, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        # print('b_s:', b_s)
        # print('b_a:', b_a)
        # print('b_r:', b_r)
        # print('q_eval:', q_eval)
        # print('q_next:', q_next)
        # print('q_target:', q_target)
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
    if dqn_model.memory_counter > dqn_model.memory_capacity:
        print(i)
        dqn_model.learn()

# a_tmp = dqn_model.choose_action(s_tmp)
# s_tmp_ = dqn_model.env_feedback(s_tmp, a_tmp)
# r_tmp = dqn_model.reward(s_tmp, s_tmp_)
# dqn_model.store_transition(s_tmp, a_tmp, r_tmp, s_tmp_)
# s_tmp = s_tmp_.clone()
# dqn_model.learn()

print('end')
