#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd
import time
import matplotlib.pyplot as plt

# Define Object and Initial Geometry
obj = pd.DataFrame(np.array([2.65, 1.95]).reshape(1, 2), columns=['R', 'r'])
init = pd.DataFrame(np.array([3.35, 2.75]).reshape(1, 2), columns=['R', 'r'])


# Obtain the Absolute Value of S Parameter
def S(s_parameter):
    s_abs = np.zeros([s_parameter.shape[0], s_parameter.shape[1], 2])
    for i in range(s_parameter.shape[0]):
        temp = np.zeros([s_parameter.shape[1], 2])
        for j in range(s_parameter.shape[1]):
            temp[j][0] = abs(complex(s_parameter[i][:, 1][j], s_parameter[i][:, 2][j]))
            temp[j][1] = abs(complex(s_parameter[i][:, 3][j], s_parameter[i][:, 4][j]))
        s_abs[i] = temp
    return s_abs


# Get the Geometry and Spectrum Data
spectrum = np.load(file='HYB_s.npy')
geometry = np.load(file='HYB_g.npy')
spectrum = S(spectrum)
geometry = geometry[:, 0: 2]
geo_table = pd.DataFrame(geometry, columns=['R', 'r'])
obj_index = geo_table[(geo_table.R == obj.loc[:, 'R'][0]) & (geo_table.r == obj.loc[:, 'r'][0])].index[0]
obj_spec = spectrum[obj_index]
init_index = geo_table[(geo_table.R == init.loc[:, 'R'][0]) & (geo_table.r == init.loc[:, 'r'][0])].index[0]
init_spec = spectrum[init_index]


# Calculate the Spectrum MSE of the two States
def spec_error(obj_spec, current_spec):
    obj_spec = np.concatenate((obj_spec[:, 0], obj_spec[:, 1]), axis=0)
    current_spec = np.concatenate((current_spec[:, 0], current_spec[:, 1]), axis=0)
    spec_error = ((obj_spec - current_spec) ** 2).sum() / len(obj_spec)
    return spec_error


# Define Hyperparameters
n_states = len(geometry)
actions = ['R+0.05', 'R-0.05', 'r+0.05', 'r-0.05']
epsilon = 0.9
alpha = 0.1
gamma = 0.9


# Build a new blank Q Table
def q_table_init(n_states, actions):
    q_table = pd.DataFrame(np.zeros((n_states, len(actions))), columns=actions)
    return q_table


# Choose the Action in each iteration
def choose_action(current_state, q_table):
    current_state_index = geo_table[(geo_table.R == current_state.loc[:, 'R'][0]) & (geo_table.r == current_state.loc[:, 'r'][0])].index[0]
    state_actions = q_table.iloc[current_state_index, :]
    if (np.random.uniform() > epsilon) or (state_actions.all() == 0):
        action_name = np.random.choice(actions)
    else:
        # action_name = state_actions.argmax()
        action_name = state_actions.idxmax()
    return action_name


# Judge the Rationality of States Changes
def para_restrict(tmp_state):
    if (tmp_state.iloc[:, 0][0] > tmp_state.iloc[:, 1][0]) & (tmp_state.iloc[:, 0][0] <= 3.80) & (tmp_state.iloc[:, 1][0] >= 1.0):
        return True
    else:
        return False


def env_feedback(current_state, action_name):
    tmp_state = pd.DataFrame.copy(current_state)
    if action_name == 'R+0.05':
        tmp_state.iloc[:, 0] = round(current_state.iloc[:, 0][0] + 0.05, 2)
        if not para_restrict(tmp_state):
            tmp_state = pd.DataFrame.copy(current_state)
            r = -1
            return tmp_state, r
    elif action_name == 'R-0.05':
        tmp_state.iloc[:, 0] = round(current_state.iloc[:, 0][0] - 0.05, 2)
        if not para_restrict(tmp_state):
            tmp_state = pd.DataFrame.copy(current_state)
            r = -1
            return tmp_state, r
    elif action_name == 'r+0.05':
        tmp_state.iloc[:, 1] = round(current_state.iloc[:, 1][0] + 0.05, 2)
        if not para_restrict(tmp_state):
            tmp_state = pd.DataFrame.copy(current_state)
            r = -1
            return tmp_state, r
    else:
        tmp_state.iloc[:, 1] = round(current_state.iloc[:, 1][0] - 0.05, 2)
        if not para_restrict(tmp_state):
            tmp_state = pd.DataFrame.copy(current_state)
            r = -1
            return tmp_state, r
    r = reward(tmp_state)
    return tmp_state, r



def reward(current_state):
    current_state_index = geo_table[(geo_table.R == current_state.loc[:, 'R'][0]) & (geo_table.r == current_state.loc[:, 'r'][0])].index[0]
    current_spec = spectrum[current_state_index]
    tmp_spec_error = spec_error(obj_spec, current_spec)
    reward = -np.log(tmp_spec_error) - 10
    # reward = tmp_spec_error
    return reward


def main():
    global q_table
    q_table = q_table_init(n_states, actions)
    # step_counter = 0
    tmp = init
    while 1:
        action_name = choose_action(tmp, q_table)
        tmp_index = geo_table[(geo_table.R == tmp.loc[:, 'R'][0]) & (geo_table.r == tmp.loc[:, 'r'][0])].index[0]
        tmp_, r = env_feedback(tmp, action_name)
        if r > 5:
            # print(tmp_)
            break
        tmp_index_ = geo_table[(geo_table.R == tmp_.loc[:, 'R'][0]) & (geo_table.r == tmp_.loc[:, 'r'][0])].index[0]
        q_predict = q_table.loc[tmp_index, action_name]
        q_target = r + gamma * q_table.iloc[tmp_index_, :].max()
        q_table.loc[tmp_index, action_name] += alpha * (q_target - q_predict)
        tmp = tmp_
        # step_counter += 1
        # if step_counter % 2000 == 0:
        #     tmpidxmax = q_table.stack().idxmax()[0]
        #     tmpargxmax = geo_table.iloc[tmpidxmax]
        #     tmp = pd.DataFrame(np.array([tmpargxmax.iloc[0], tmpargxmax.iloc[1]]).reshape(1, 2), columns=['R', 'r'])
        #     q_table = q_table_init(n_states, actions)
        # if step_counter % 1 == 0:
            # print('iteration:', step_counter)
            # print(tmp)


def counter():
    a = pd.DataFrame.copy(q_table)
    a[a!=0] = 1
    count = 0
    for i in range(len(a)):
        aa = a.iloc[i, :]
        if aa[0] + aa[1] + aa[2] + aa[3] > 0:
            count += 1
    return count


if __name__ == '__main__':
    cc = []
    epoch = 0
    for ii in range(100):
        main()
        nn = counter()
        cc.append(nn)
        epoch += 1
        print(epoch)
    means = sum(cc) / len(cc)
    print(means)
    print('end')


