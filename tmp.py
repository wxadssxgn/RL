# -*- coding:utf-8 -*-

import numpy as np
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


print('--end--')
