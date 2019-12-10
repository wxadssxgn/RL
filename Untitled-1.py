#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

memory = np.load('HYB_memory.npy')

obj = np.array([2.65, 1.95]).reshape(1, 2)
init = np.array([3.00, 2.20]).reshape(1, 2)

class model(nn.Module):
    def __init__(self, ):
        super(model, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(2, 10), nn.Dropout(0), nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Linear(10, 40), nn.Dropout(0), nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.Linear(40, 100), nn.Dropout(0), nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(nn.Linear(100, 100), nn.Dropout(0), nn.ReLU(inplace=True))
        self.layer5 = nn.Sequential(nn.Linear(100, 40), nn.Dropout(0), nn.ReLU(inplace=True))
        self.layer6 = nn.Sequential(nn.Linear(40, 10), nn.Dropout(0), nn.ReLU(inplace=True))
        self.layer7 = nn.Sequential(nn.Linear(10, 4))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        return x






print('--end--')
