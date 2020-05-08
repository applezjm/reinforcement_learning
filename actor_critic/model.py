#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return super().log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

class AC(nn.Module):
    # Advantage Actor-Critic
    def __init__(self, input_shape, action_dim, use_batchnorm=False):
        # 调用父类构造方法
        super(AC, self).__init__()
        # CNN extract feature
        if use_batchnorm:
            self.conv = nn.Sequential(
                nn.Conv2d(input_shape[0], 32, 8, 4), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 32, 3, 1), nn.BatchNorm2d(32), nn.ReLU()
            )
            conv_output_size = self._get_conv_out(input_shape)
            self.flatten = nn.Sequential(
                nn.Linear(conv_output_size, 512), nn.BatchNorm1d(512), nn.ReLU()
            )
        else:
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
        
            self.conv = nn.Sequential(
                init_(nn.Conv2d(input_shape[0], 32, 8, 4)), nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, 2)), nn.ReLU(),
                init_(nn.Conv2d(64, 32, 3, 1)), nn.ReLU()
            )
            conv_output_size = self._get_conv_out(input_shape)
            self.flatten = nn.Sequential(
                init_(nn.Linear(conv_output_size, 512)), nn.ReLU()
            )

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)
        
        self.actor = nn.Sequential(
            init_(nn.Linear(512, action_dim)) 
        )
        
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        
        self.critic = nn.Sequential(
            init_(nn.Linear(512, 1))
        )
    
    def _get_conv_out(self, input_shape):
        fake_output = self.conv(torch.zeros(1, *input_shape))
        return int(np.prod(fake_output.size()))

    def forward(self, x):
        conv_output = self.flatten(self.conv(x / 255.0).view(x.size(0), -1))
        return FixedCategorical(logits=self.actor(conv_output)), self.critic(conv_output)

