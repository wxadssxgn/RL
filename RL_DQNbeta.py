#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

spectrum = np.load(file='HYB_s.npy')
geometry = np.load(file='HYB_g.npy')[:, 0: 2]


print('end')
