#! /usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np

a = np.load('ring100.npy')

fig = plt.figure(figsize=(9,5))
x = np.arange(1, 101)
plt.plot(x, a, linewidth=2.5 ,color='black')
plt.grid(True, linestyle='-.')
plt.xlim((0, 100))
# plt.yscale('log')
plt.xlabel('Iterations', size=16)
plt.ylabel('Steps', size=16)
plt.show()

print('end')
