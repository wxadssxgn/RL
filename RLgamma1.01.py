#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# Define Object and Initial Geometry
obj = pd.DataFrame(np.array([7.2, 3.4, 2.8, 1.3]).reshape(1, 4), columns=['SubWidth', 'RadiusOuter', 'RadiusInner', 'GapLength'])
init = pd.DataFrame(np.array([8.0, 2.8, 1.8, 0.7]).reshape(1, 4), columns=['SubWidth', 'RadiusOuter', 'RadiusInner', 'GapLength'])

# Obtain the Absolute Value of S Parameter
def S(s_parameter):
    s_abs = np.zeros([s_parameter.shape[0], s_parameter.shape[1], 2])
    for i in range(s_parameter.shape[0]):
        temp = np.zeros([s_parameter.shape[1], 2])
        for j in range(s_parameter.shape[1]):
            temp[j][0] = abs(complex(s_parameter[i][:, 0][j], s_parameter[i][:, 1][j]))
            temp[j][1] = abs(complex(s_parameter[i][:, 2][j], s_parameter[i][:, 3][j]))
        s_abs[i] = temp
    return s_abs

# Get the Geometry and Spectrum Data
spectrum = np.load(file='spec_fss_gap_ring.npy')
geometry = np.load(file='geo_fss_gap_ring.npy')
spectrum = S(spectrum)
geometry = geometry[:, 1: ]
geo_table = pd.DataFrame(geometry, columns=['SubWidth', 'RadiusOuter', 'RadiusInner', 'GapLength'])
obj_index = geo_table[(geo_table.SubWidth == obj.loc[:, 'SubWidth'][0]) & (geo_table.RadiusOuter == obj.loc[:, 'RadiusOuter'][0]) \
    & (geo_table.RadiusInner == obj.loc[:, 'RadiusInner'][0]) & (geo_table.GapLength == obj.loc[:, 'GapLength'][0])].index[0]
obj_spec = spectrum[obj_index]
init_index = geo_table[(geo_table.SubWidth == init.loc[:, 'SubWidth'][0]) & (geo_table.RadiusOuter == init.loc[:, 'RadiusOuter'][0]) \
    & (geo_table.RadiusInner == init.loc[:, 'RadiusInner'][0]) & (geo_table.GapLength == init.loc[:, 'GapLength'][0])].index[0]
init_spec = spectrum[init_index]

# Calculate the Spectrum MSE of the two States
def spec_error(obj_spec, current_spec):
    obj_spec = np.concatenate((obj_spec[:, 0], obj_spec[:, 1]), axis=0)
    current_spec = np.concatenate((current_spec[:, 0], current_spec[:, 1]), axis=0)
    spec_error = ((obj_spec - current_spec) ** 2).sum() / len(obj_spec)
    return spec_error

# Define Hyperparameters
n_states = len(geometry)
actions = ['SubWidth+0.2', 'SubWidth-0.2', 'RadiusOuter+0.2', 'RadiusOuter-0.2', \
    'RadiusInner+0.2', 'RadiusInner-0.2', 'GapLength+0.2', 'GapLength-0.2']
epsilon = 0.9
alpha = 0.1
gamma = 0.9
max_episode = 500

# Build a new blank Q Table
def q_table_init(n_states, actions):
    q_table = pd.DataFrame(np.zeros((n_states, len(actions))), columns=actions)
    return q_table

q_table = q_table_init(n_states, actions)

# Choose the Action in each iteration
def choose_action(current_state, q_table):
    current_state_index = geo_table[(geo_table.SubWidth == current_state.loc[:, 'SubWidth'][0]) & (geo_table.RadiusOuter \
        == current_state.loc[:, 'RadiusOuter'][0]) & (geo_table.RadiusInner == current_state.loc[:, 'RadiusInner'][0]) \
            & (geo_table.GapLength == current_state.loc[:, 'GapLength'][0])].index[0]
    state_actions = q_table.iloc[current_state_index, :]
    if (np.random.uniform() > epsilon) or (state_actions.all() == 0):
        action_name = np.random.choice(actions)
    else:
        action_name = state_actions.idxmax()
    return action_name

# Judge the Rationality of States Changes
def para_restrict(tmp_state):
    # SubWidth > 2*RingRadius_Outer
    # RingRadius_Outer > RingRadius_Inner
    # 2*RingRadius_Inner > GapLength
    # SubThickness: [0.1, 0.2]
    # SubWidth: [7, 9] step: 0.2
    # RingRadius_Outer: [1.4, 4.2] step: 0.2
    # RingRadius_Inner: [1.0, 3.6] step: 0.2
    # GapLength: [0.5, 5.5] step: 0.2
    # RingHeight = 0.05
    # SubWidth RadiusOuter RadiusInner GapLength
    if (tmp_state.loc[:, 'SubWidth'][0] > 2*tmp_state.loc[:, 'RadiusOuter'][0]) & \
        (tmp_state.loc[:, 'RadiusOuter'][0] > tmp_state.loc[:, 'RadiusInner'][0]) & \
            (2*tmp_state.loc[:, 'RadiusInner'][0] > tmp_state.loc[:, 'GapLength'][0]) & \
                (tmp_state.loc[:, 'SubWidth'][0] >= 7) & (tmp_state.loc[:, 'SubWidth'][0] <= 9) & \
                    (tmp_state.loc[:, 'RadiusOuter'][0] >= 1.4) & (tmp_state.loc[:, 'RadiusOuter'][0] <= 4.2) & \
                        (tmp_state.loc[:, 'RadiusInner'][0] >= 1.0) & (tmp_state.loc[:, 'RadiusInner'][0] <= 3.6) & \
                            (tmp_state.loc[:, 'GapLength'][0] >= 0.5) & (tmp_state.loc[:, 'GapLength'][0] <= 5.5):
        return True
    else:
        return False

# 'SubWidth+0.2', 'SubWidth-0.2', 'RadiusOuter+0.2', 'RadiusOuter-0.2'
# 'RadiusInner+0.2', 'RadiusInner-0.2', 'GapLength+0.2', 'GapLength-0.2'
def env_feedback(current_state, action_name):
    tmp_state = pd.DataFrame.copy(current_state)
    if action_name == 'SubWidth+0.2':
        tmp_state.loc[:, 'SubWidth'] = round(current_state.loc[:, 'SubWidth'][0] + 0.2, 2)
        if not para_restrict(tmp_state):
            tmp_state = pd.DataFrame.copy(current_state)
            r = -10
            return tmp_state, r
    elif action_name == 'SubWidth-0.2':
        tmp_state.loc[:, 'SubWidth'] = round(current_state.loc[:, 'SubWidth'][0] - 0.2, 2)
        if not para_restrict(tmp_state):
            tmp_state = pd.DataFrame.copy(current_state)
            r = -10
            return tmp_state, r
    elif action_name == 'RadiusOuter+0.2':
        tmp_state.loc[:, 'RadiusOuter'] = round(current_state.loc[:, 'RadiusOuter'][0] + 0.2, 2)
        if not para_restrict(tmp_state):
            tmp_state = pd.DataFrame.copy(current_state)
            r = -10
            return tmp_state, r
    elif action_name == 'RadiusOuter-0.2':
        tmp_state.loc[:, 'RadiusOuter'] = round(current_state.loc[:, 'RadiusOuter'][0] - 0.2, 2)
        if not para_restrict(tmp_state):
            tmp_state = pd.DataFrame.copy(current_state)
            r = -10
            return tmp_state, r
    elif action_name == 'RadiusInner+0.2':
        tmp_state.loc[:, 'RadiusInner'] = round(current_state.loc[:, 'RadiusInner'][0] + 0.2, 2)
        if not para_restrict(tmp_state):
            tmp_state = pd.DataFrame.copy(current_state)
            r = -10
            return tmp_state, r
    elif action_name == 'RadiusInner-0.2':
        tmp_state.loc[:, 'RadiusInner'] = round(current_state.loc[:, 'RadiusInner'][0] - 0.2, 2)
        if not para_restrict(tmp_state):
            tmp_state = pd.DataFrame.copy(current_state)
            r = -10
            return tmp_state, r
    elif action_name == 'GapLength+0.2':
        tmp_state.loc[:, 'GapLength'] = round(current_state.loc[:, 'GapLength'][0] + 0.2, 2)
        if not para_restrict(tmp_state):
            tmp_state = pd.DataFrame.copy(current_state)
            r = -10
            return tmp_state, r
    elif action_name == 'GapLength-0.2':
        tmp_state.loc[:, 'GapLength'] = round(current_state.loc[:, 'GapLength'][0] - 0.2, 2)
        if not para_restrict(tmp_state):
            tmp_state = pd.DataFrame.copy(current_state)
            r = -10
            return tmp_state, r
    r = reward(tmp_state)
    return tmp_state, r

def reward(current_state):
    current_state_index = geo_table[(geo_table.SubWidth == current_state.loc[:, 'SubWidth'][0]) & (geo_table.RadiusOuter \
        == current_state.loc[:, 'RadiusOuter'][0]) & (geo_table.RadiusInner == current_state.loc[:, 'RadiusInner'][0]) \
            & (geo_table.GapLength == current_state.loc[:, 'GapLength'][0])].index[0]
    current_spec = spectrum[current_state_index]
    tmp_spec_error = spec_error(obj_spec, current_spec)
    reward = -np.log(tmp_spec_error) - 10
    # reward = tmp_spec_error
    return reward

steps_list= []
def main():
    for episode in range(max_episode):
        # tmp: TEMP STATE
        # tmp_: NEXT STATE
        tmp = init
        step_counter = 0
        while 1:
            action_name = choose_action(tmp, q_table)
            tmp_index = geo_table[(geo_table.SubWidth == tmp.loc[:, 'SubWidth'][0]) & (geo_table.RadiusOuter == \
                tmp.loc[:, 'RadiusOuter'][0]) & (geo_table.RadiusInner == tmp.loc[:, 'RadiusInner'][0]) & (geo_table.GapLength \
                    == tmp.loc[:, 'GapLength'][0])].index[0]
            tmp_, r = env_feedback(tmp, action_name)
            if r > 5:
                q_table.loc[tmp_index, action_name] += 1e2
                step_counter += 1
                break
            tmp_index_ = geo_table[(geo_table.SubWidth == tmp_.loc[:, 'SubWidth'][0]) & (geo_table.RadiusOuter == \
                tmp_.loc[:, 'RadiusOuter'][0]) & (geo_table.RadiusInner == tmp_.loc[:, 'RadiusInner'][0]) & (geo_table.GapLength \
                    == tmp_.loc[:, 'GapLength'][0])].index[0]
            q_predict = q_table.loc[tmp_index, action_name]
            q_target = r + gamma * q_table.iloc[tmp_index_, :].max()
            q_table.loc[tmp_index, action_name] += alpha * (q_target - q_predict)
            tmp = tmp_
            step_counter += 1
        steps_list.append(step_counter)
        print('episode: ', episode+1, '  ', 'steps: ',  step_counter)

if __name__ == '__main__':
    main()

    print('end')
