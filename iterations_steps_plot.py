#! /usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

a = np.load('500_iteration_init_3.35_2.75_obj_2.95_2.25.npy')

fig = plt.figure(figsize=(9,5))
plt.plot(a, color='black')
plt.grid(True, linestyle='-.')
plt.xlim((0, 500))
plt.yscale('log')
plt.xlabel('Iterations', size=16)
plt.ylabel('Steps', size=16)
plt.show()

print('end')
