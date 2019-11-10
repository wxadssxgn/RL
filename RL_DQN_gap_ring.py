#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
import matplotlib.pyplot as plt

class model(nn.Module):
    def __init__(self, ):
        super(model, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(4, 10), nn.Dropout(0), nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Linear(10, 40), nn.Dropout(0), nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.Linear(40, 100), nn.Dropout(0), nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(nn.Linear(100, 100), nn.Dropout(0), nn.ReLU(inplace=True))
        self.layer5 = nn.Sequential(nn.Linear(100, 100), nn.Dropout(0), nn.ReLU(inplace=True))
        self.layer6 = nn.Sequential(nn.Linear(100, 40), nn.Dropout(0), nn.ReLU(inplace=True))
        self.layer7 = nn.Sequential(nn.Linear(40, 10), nn.Dropout(0), nn.ReLU(inplace=True))
        self.layer8 = nn.Sequential(nn.Linear(10, 8))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x


class DQN():
    def __init__(self, ):
        spectrum = np.load(file='spec_fss_gap_ring.npy')
        self.spectrum = self.get_s_para(spectrum)
        geometry = np.load(file='geo_fss_gap_ring.npy')
        self.geometry = geometry[:, 1: ]
        obj_index = self.get_index(obj)
        init_index = self.get_index(init)
        self.obj_spec = self.spectrum[obj_index]
        self.init_spec = self.spectrum[init_index]
        self.n_states = 4
        self.actions = ['SubWidth+0.2', 'SubWidth-0.2', 'RadiusOuter+0.2', 'RadiusOuter-0.2', \
                        'RadiusInner+0.2', 'RadiusInner-0.2', 'GapLength+0.2', 'GapLength-0.2']
        self.epsilon = 0.9
        self.greedy_epsilon = 1
        self.alpha = 0.1
        self.gamma = 0.9
        self.memory_capacity = 500
        self.LR = 1e-4
        self.batch_size = 300

        self.eval_model = model().cuda() if torch.cuda.is_available() else model()
        self.target_model = model().cuda() if torch.cuda.is_available() else model()
        self.eval_model.apply(self.weights_init_uniform)
        self.target_model.apply(self.weights_init_uniform)

        self.learn_step_counter = 0
        self.memory_counter = 0
        self.target_replace_iteration = 30
        self.memory = np.zeros((self.memory_capacity, 1, self.n_states * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_model.parameters(), lr=self.LR)
        self.loss_func = nn.MSELoss()
        self.loss_list = []
        self.loss_tmp = torch.cuda.FloatTensor([0]) if torch.cuda.is_available() else torch.Tensor([0])

    def weights_init_uniform(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            m.weight.data.uniform_(0.0, 0.1)
            m.bias.data.fill_(0)

    def get_index(self, s):
        for i in range(len(self.geometry)):
            if (self.geometry[i] == np.array(s).astype(float).round(2).reshape(1, 4)).all():
                return i

    def get_s_para(self, s_parameter):
        s_abs = np.zeros([s_parameter.shape[0], s_parameter.shape[1], 2])
        for i in range(s_parameter.shape[0]):
            temp = np.zeros([s_parameter.shape[1], 2])
            for j in range(s_parameter.shape[1]):
                temp[j][0] = abs(complex(s_parameter[i][:, 0][j], s_parameter[i][:, 1][j]))
                temp[j][1] = abs(complex(s_parameter[i][:, 2][j], s_parameter[i][:, 3][j]))
            s_abs[i] = temp
        return s_abs
    
    def choose_action(self, s):
        # a_index.type = array
        if np.random.uniform() < self.epsilon:
            action_values = self.eval_model.forward(s)
            a_index = torch.max(action_values, 1)[1].cpu().numpy()
        else:
            a_index = np.random.randint(0, 8, size=(1, ))
        return a_index

    def greedy_choose_action(self, s, i):
        # a_index.type = array
        if np.random.uniform() > self.greedy_epsilon:
            action_values = self.eval_model.forward(s)
            a_index = torch.max(action_values, 1)[1].cpu().numpy()
        else:
            a_index = np.random.randint(0, 8, size=(1, ))
        if i % 500 == 0:
            self.greedy_epsilon = self.greedy_epsilon * 0.9
        if self.greedy_epsilon < 0.1:
            self.greedy_epsilon = 0.1
        return a_index
    
    def env_feedback(self, s, a):
        # s.type = Tensor
        # s.shape = [SubWidth, RadiusOuter, RadiusInner, GapLength]
        # SubWidth > 2*RingRadius_Outer
        # RingRadius_Outer > RingRadius_Inner
        # 2*RingRadius_Inner > GapLength
        # SubThickness: [0.1, 0.2]
        # SubWidth: [7, 9] step: 0.2
        # RingRadius_Outer: [1.4, 4.2] step: 0.2
        # RingRadius_Inner: [1.0, 3.6] step: 0.2
        # GapLength: [0.5, 5.5] step: 0.2
        # RingHeight = 0.05
        # s[0, 0] = SubWidth
        # s[0, 1] = RadiusOuter
        # s[0, 2] = RadiusInner
        # s[0, 3] = GapLength
        s_ = s.clone()
        if a == 0:
            s_[0, 0] = s_[0, 0] + 0.2
        elif a == 1:
            s_[0, 0] = s_[0, 0] - 0.2
        elif a == 2:
            s_[0, 1] = s_[0, 1] + 0.2
        elif a == 3:
            s_[0, 1] = s_[0, 1] - 0.2
        elif a == 4:
            s_[0, 2] = s_[0, 2] + 0.2
        elif a == 5:
            s_[0, 2] = s_[0, 2] - 0.2
        elif a == 6:
            s_[0, 3] = s_[0, 3] + 0.2
        else:
            s_[0, 3] = s_[0, 3] - 0.2
        tmp_s = s_.cpu().numpy().round(2)
        if tmp_s[0, 0] > 9.00 or tmp_s[0, 0] < 7.00 or tmp_s[0, 1] > 4.20 or tmp_s[0, 1] < 1.40 \
           or tmp_s[0, 2] > 3.60 or tmp_s[0, 2] < 1.00 or tmp_s[0, 3] > 5.50 or tmp_s[0, 3] < 0.5 \
           or tmp_s[0, 0] <= 2*tmp_s[0, 1] or tmp_s[0, 1] <= tmp_s[0, 2] or 2*tmp_s[0, 2] \
           <= tmp_s[0, 3]:
            s_ = s.clone()
        return s_

    def replay_buffer(self, s, a, r, s_):
        s = s.cpu()
        s_ = s_.cpu()
        s = np.array(s).astype(float).round(2).reshape(1, 4)
        s_ = np.array(s_).astype(float).round(2).reshape(1, 4)
        a = np.array(a).astype(int).reshape(1, 1)
        r = np.array(r).astype(float).reshape(1, 1)
        transition = np.hstack((s, a, r, s_))
        index = self.memory_counter % self.memory_capacity
        self.memory[index] = transition
        self.memory_counter += 1

    def reward(self, s, s_):
        s = s.cpu()
        s_ = s_.cpu()
        s_index = self.get_index(s)
        s_index_ = self.get_index(s_)
        if s_index != s_index_:
            s_spec = self.spectrum[s_index]
            s_spec_ = self.spectrum[s_index_]
            s_spec = np.concatenate((s_spec[:, 0], s_spec[:, 1]), axis=0)
            s_spec_ = np.concatenate((s_spec_[:, 0], s_spec_[:, 1]), axis=0)
            s_error = ((np.concatenate((self.obj_spec[:, 0], self.obj_spec[:, 1]), axis=0) - s_spec) ** 2).sum() / len(self.obj_spec)
            s_error_ = ((np.concatenate((self.obj_spec[:, 0], self.obj_spec[:, 1]), axis=0) - s_spec_) ** 2).sum() / len(self.obj_spec)
            r1 = -np.log(max(s_error, 1e-4))
            r2 = -np.log(max(s_error_, 1e-4))
            r = (r2 - r1) * 15
            # r = -np.log(max(s_error_, 1e-4)) - 3
            return r
        else:
            r = -5
            return r

    def learn(self):
        if self.learn_step_counter % self.target_replace_iteration == 0:
            self.target_model.load_state_dict(self.eval_model.state_dict())
        self.learn_step_counter += 1
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        memory_tmp = self.memory[sample_index]
        if torch.cuda.is_available():
            s_tmp = torch.cuda.FloatTensor(memory_tmp[:, :, :self.n_states])
            a_tmp = torch.cuda.LongTensor(memory_tmp[:, :, self.n_states: self.n_states+1].astype(int))
            r_tmp = torch.cuda.FloatTensor(memory_tmp[:, :, self.n_states+1: self.n_states+2])
            s_tmp_ = torch.cuda.FloatTensor(memory_tmp[:, :, -self.n_states :])
        else:
            s_tmp = torch.FloatTensor(memory_tmp[:, :, :self.n_states])
            a_tmp = torch.LongTensor(memory_tmp[:, :, self.n_states: self.n_states+1].astype(int))
            r_tmp = torch.FloatTensor(memory_tmp[:, :, self.n_states+1: self.n_states+2])
            s_tmp_ = torch.FloatTensor(memory_tmp[:, :, -self.n_states :])
        q_eval = self.eval_model(s_tmp).gather(2, a_tmp).reshape(self.batch_size, 1)
        q_next = self.target_model(s_tmp_).detach()     # detach from graph, don't backpropagate
        q_target = r_tmp.reshape(self.batch_size, 1) + self.gamma * q_next.max(2)[0].view(self.batch_size, 1)   # shape (batch, 1)
        self.loss_tmp = self.loss_func(q_eval, q_target)
        # print('loss:', self.loss_tmp)
        self.loss_list.append(self.loss_tmp.item())
        self.optimizer.zero_grad()
        self.loss_tmp.backward()
        self.optimizer.step()


if __name__ == '__main__':
    # Define object and initial geometry
    obj = np.array([7.2, 3.4, 2.8, 1.3]).reshape(1, 4)
    init = np.array([8.0, 2.8, 1.8, 0.7]).reshape(1, 4)
    dqn = DQN()
    s_tmp = torch.cuda.FloatTensor(init) if torch.cuda.is_available() else torch.Tensor(init)

    for i in range(1, 10001):
        # a_tmp = dqn.choose_action(s_tmp)
        a_tmp = dqn.greedy_choose_action(s_tmp, i)
        s_tmp_ = dqn.env_feedback(s_tmp, a_tmp)
        r_tmp = dqn.reward(s_tmp, s_tmp_)
        dqn.replay_buffer(s_tmp, a_tmp, r_tmp, s_tmp_)
        if r_tmp == -5:
            s_tmp = torch.cuda.FloatTensor(init) if torch.cuda.is_available() else torch.Tensor(init)
            dqn.greedy_epsilon = 1
            a_tmp = dqn.choose_action(s_tmp)
            s_tmp_ = dqn.env_feedback(s_tmp, a_tmp)
            r_tmp = dqn.reward(s_tmp, s_tmp_)
        dqn.replay_buffer(s_tmp, a_tmp, r_tmp, s_tmp_)
        s_tmp = s_tmp_.clone()
        if dqn.memory_counter > dqn.memory_capacity:
            dqn.learn()
        if i % 1000 == 0:
            print(dqn.loss_tmp)

    loss_list = dqn.loss_list
    plt.plot(loss_list)
    plt.show()


    print('end')
