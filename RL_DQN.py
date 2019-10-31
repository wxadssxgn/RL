#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
import matplotlib.pyplot as plt

# Define Object and Initial Geometry
OBJ = pd.DataFrame(np.array([2.65, 1.95]).reshape(1, 2), columns=['R', 'r'])
INIT = pd.DataFrame(np.array([3.35, 2.75]).reshape(1, 2), columns=['R', 'r'])

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

# Get the Geometry and Spectrum Data
SPECTRUM = np.load(file='HYB_s.npy')
SPECTRUM = Get_S_Para(SPECTRUM)
GEOMETRY = np.load(file='HYB_g.npy')[:, 0: 2]
GEO_TABLE = pd.DataFrame(GEOMETRY, columns=['R', 'r'])
OBJ_INDEX = GEO_TABLE[(GEO_TABLE.R==OBJ.loc[:, 'R'][0])&(GEO_TABLE.r==OBJ.loc[:, 'r'][0])].index[0]
OBJ_SPEC = SPECTRUM[OBJ_INDEX]
INIT_INDEX = GEO_TABLE[(GEO_TABLE.R==INIT.loc[:, 'R'][0])&(GEO_TABLE.r==INIT.loc[:, 'r'][0])].index[0]
INIT_SPEC = SPECTRUM[INIT_INDEX]

# Define Hyperparameters
N_STATES = len(GEOMETRY)
ACTIONS = ['R+0.05', 'R-0.05', 'r+0.05', 'r-0.05']
EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9

class MODEL(nn.Module):
    def __init__(self, ):
        super(MODEL, self).__init__()
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

class DQN():
    def __init__(self, ):
        self.EVAL_MODEL = MODEL()
        self.TAR_MODEL = MODEL()
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def CHOOSE_ACTION(self, x):
        if np.random.uniform() < EPSILON:
            ACTION = self.EVAL_MODEL.forward(x)
            return ACTION
    
    def MEMORY(self, STATE, ACTION, REWARD, STATE_):
        transition = np.hstack(STATE, (ACTION, REWARD), STATE_)
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def LEARN(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()

print('\nCollecting experience...')

for i_episode in range(400):
    s = env.reset()
    ep_r = 0
    while True:
        env.render()
        a = dqn.choose_action(s)
        # take action
        s_, r, done, info = env.step(a)
        # modify the reward
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2
        dqn.store_transition(s, a, r, s_)
        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))
        if done:
            break
        s = s_




print('end')
