#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

class A2C(nn.Module):
    # Advantage Actor-Critic
    def __init__(self, input_shape, action_dim):
        # 调用父类构造方法
        super(A2C, self).__init__()
        # CNN extract feature
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 2, 1),
            nn.ReLU()
        )
        conv_output_size = self._get_conv_out(input_shape)
       
        self.actor = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
            nn.Softmax(dim=1)
        )
        self.critic = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def _get_conv_out(self, input_shape):
        fake_output = self.conv(torch.zeros(1, *input_shape))
        return int(np.prod(fake_output.size()))

    def forward(self, x):
        conv_output = self.conv(x).view(x.size()[0], -1)
        return self.actor(conv_output), self.critic(conv_output)
