#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd
import time
import matplotlib.pyplot as plt

# Define Object and Initial Geometry
obj = pd.DataFrame(np.array([2.95, 2.25]).reshape(1, 2), columns=['R', 'r'])
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
freq = spectrum[0, :, 0]
geometry = np.load(file='HYB_g.npy')
spectrum = S(spectrum)
geometry = geometry[:, 0: 2]
geo_table = pd.DataFrame(geometry, columns=['R', 'r'])
obj_index = geo_table[(geo_table.R == obj.loc[:, 'R'][0]) & (geo_table.r == obj.loc[:, 'r'][0])].index[0]
obj_spec = spectrum[obj_index]
init_index = geo_table[(geo_table.R == init.loc[:, 'R'][0]) & (geo_table.r == init.loc[:, 'r'][0])].index[0]
init_spec = spectrum[init_index]

# Define requirements
demand_index = np.where(obj_spec[:, 0]**2 > 0.9)[0]

demand_line = np.zeros((1, len(freq)))
for i in demand_index:
    demand_line[0, i] = 0.95
demand_line = np.squeeze(demand_line)

# Calculate the error between current state and demanding spectrum
def get_error(current_spec):
    current_spec = current_spec[:, 0]
    current_spec_tmp = current_spec[demand_index[0]: demand_index[-1]+1]
    spec_error = []
    for i in range(len(current_spec_tmp)):
        if current_spec_tmp[i]**2 < 0.9:
            error_tmp = 0.9 - current_spec_tmp[i]**2
            spec_error.append(error_tmp)
    spec_error = sum(spec_error)
    return spec_error

# Define Hyperparameters
n_states = len(geometry)
actions = ['R+0.05', 'R-0.05', 'r+0.05', 'r-0.05']
epsilon = 0.9
alpha = 0.1
gamma = 0.9
# max_episode = 500

# Build a new blank Q Table
def q_table_init(n_states, actions):
    q_table = pd.DataFrame(np.zeros((n_states, len(actions))), columns=actions)
    return q_table

q_table = q_table_init(n_states, actions)

# Choose the Action in each iteration
def choose_action(current_state, q_table):
    current_state_index = geo_table[(geo_table.R == current_state.loc[:, 'R'][0]) & (geo_table.r == current_state.loc[:, 'r'][0])].index[0]
    state_actions = q_table.iloc[current_state_index, :]
    if (np.random.uniform() > epsilon) or (state_actions.all() == 0):
        action_name = np.random.choice(actions)
    else:
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
            r = -10
            return tmp_state, r
    elif action_name == 'R-0.05':
        tmp_state.iloc[:, 0] = round(current_state.iloc[:, 0][0] - 0.05, 2)
        if not para_restrict(tmp_state):
            tmp_state = pd.DataFrame.copy(current_state)
            r = -10
            return tmp_state, r
    elif action_name == 'r+0.05':
        tmp_state.iloc[:, 1] = round(current_state.iloc[:, 1][0] + 0.05, 2)
        if not para_restrict(tmp_state):
            tmp_state = pd.DataFrame.copy(current_state)
            r = -10
            return tmp_state, r
    else:
        tmp_state.iloc[:, 1] = round(current_state.iloc[:, 1][0] - 0.05, 2)
        if not para_restrict(tmp_state):
            tmp_state = pd.DataFrame.copy(current_state)
            r = -10
            return tmp_state, r
    r = reward(tmp_state)
    return tmp_state, r

def reward(current_state):
    current_state_index = geo_table[(geo_table.R == current_state.loc[:, 'R'][0]) & (geo_table.r == current_state.loc[:, 'r'][0])].index[0]
    current_spec = spectrum[current_state_index]
    tmp_spec_error = get_error(current_spec)
    reward = -np.log(tmp_spec_error) - 5
    return reward

def main():
    step_counter = 0
    tmp = init
    while 1:
        action_name = choose_action(tmp, q_table)
        tmp_index = geo_table[(geo_table.R == tmp.loc[:, 'R'][0]) & (geo_table.r == tmp.loc[:, 'r'][0])].index[0]
        tmp_, r = env_feedback(tmp, action_name)
        if r > 5:
            q_table.loc[tmp_index, action_name] += 1e2
            global res
            res = tmp_
            step_counter += 1
            print(tmp_)
            break
        tmp_index_ = geo_table[(geo_table.R == tmp_.loc[:, 'R'][0]) & (geo_table.r == tmp_.loc[:, 'r'][0])].index[0]
        q_predict = q_table.loc[tmp_index, action_name]
        q_target = r + gamma * q_table.iloc[tmp_index_, :].max()
        q_table.loc[tmp_index, action_name] += alpha * (q_target - q_predict)
        tmp = tmp_
        step_counter += 1
        if step_counter % 1 == 0:
            print('iteration:', step_counter)
            print(tmp)


if __name__ == '__main__':
    main()
    res_index = geo_table[(geo_table.R == res.loc[:, 'R'][0]) & (geo_table.r == res.loc[:, 'r'][0])].index[0]
    res_spec = spectrum[res_index]
    fig = plt.figure(figsize=(9,5))
    plt.title('R:{0} r:{1}'.format(res.loc[:, 'R'][0], res.loc[:, 'r'][0]), fontsize=16)
    plt.plot(res_spec[:, 0], color='blue', linewidth=2.5, label='S11')
    plt.plot(res_spec[:, 1], color='red', linewidth=2.5, label='S22')
    plt.plot(demand_line, color='green', linewidth=1.5)
    plt.legend(fontsize=14, loc='best')
    plt.grid(True, linestyle='-.')
    plt.xlim((10, 30))
    plt.ylim((0, 1.05))
    plt.xlabel('Frequency/GHz', size=16)
    plt.ylabel('S-Parameters', size=16)
    plt.show()

    print('end')
