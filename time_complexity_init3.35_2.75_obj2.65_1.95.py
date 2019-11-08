#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

a = np.load('ring100_.npy')
means = sum(a) / len(a)
b = []
b.append(means)
b = b * len(a)

fig = plt.figure(figsize=(9,5))
plt.plot(a, color='blue', linewidth=2.2, label='Real-time')
plt.plot(b, color='red', linewidth=2.2, label='Mean')
plt.legend(fontsize=14, loc='best')
plt.grid(True, linestyle='-.')
plt.xlim((0, 100))
# plt.ylim((0, 1400))
plt.xlabel('Iterations', size=16)
plt.ylabel('Numbers of States Traversed', size=16)
plt.show()

print('end')