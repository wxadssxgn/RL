#! /usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

a = np.load('steps_gap_ring_q_learning.npy')

fig = plt.figure(figsize=(9,5))
x = np.arange(1, 501)
plt.plot(x, a, linewidth=1.5 ,color='black')
plt.grid(True, linestyle='-.')
# plt.xlim((0, 500))
plt.yscale('log')
plt.xlabel('Iterations', size=16)
plt.ylabel('Steps', size=16)
plt.show()

print('end')
