#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):
    def __init__(self, input_shape, action_dim):
        # 调用父类构造方法
        super(DQN, self).__init__()
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
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )
    
    def _get_conv_out(self, input_shape):
        fake_output = self.conv(torch.zeros(1, *input_shape))
        return int(np.prod(fake_output.size()))

    def forward(self, x):
        conv_output = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_output)

class DuelingDQN(nn.Module):
    def __init__(self, input_shape, action_dim):
        super(DuelingDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 2, 1),
            nn.ReLU()
        )
        conv_output_size = self._get_conv_out(input_shape)
        self.fc_value = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.fc_advantage = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def _get_conv_out(self, input_shape):
        fake_output = self.conv(torch.zeros(1, *input_shape))
        return int(np.prod(fake_output.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        value = self.fc_value(conv_out)
        advantage = self.fc_advantage(conv_out)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))

class NoisyFactorizedLayer(nn.Linear):
    def __init__(self, in_features, out_features, sigma_zero=0.4, bias=True):
        super(NoisyFactorizedLayer, self).__init__(in_features, out_features, bias=True)

        # 定义两组参数，一组是可学习的sigma，一组是不可学习的epsilon
        sigma_init = sigma_zero / math.sqrt(in_features)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))

        self.register_buffer("epsilon_input", torch.zeros(1, in_features))
        self.register_buffer("epsilon_output", torch.zeros(out_features, 1))

    def forward(self, input):
        # 先正则化，再sgn(x)*(x^1/2)
        self.epsilon_input.normal_()
        self.epsilon_output.normal_()

        norm_func = lambda x:torch.sign(x)*torch.sqrt(torch.abs(x))
        ep_in = norm_func(self.epsilon_input)
        ep_out = norm_func(self.epsilon_output)

        weight = self.weight + torch.mm(ep_out, ep_in) * self.sigma_weight
        bias = self.bias + ep_out.squeeze(-1) * self.sigma_bias
        return F.linear(input, weight, bias)

class NoisyDQN(nn.Module):
    def __init__(self, input_shape, action_dim):
        super(NoisyDQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 2, 1),
            nn.ReLU()
        )
        conv_output_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            NoisyFactorizedLayer(conv_output_size, 512),
            nn.ReLU(),
            NoisyFactorizedLayer(512, action_dim)
        )
    
    def _get_conv_out(self, input_shape):
        fake_output = self.conv(torch.zeros(1, *input_shape))
        return int(np.prod(fake_output.size()))

    def forward(self, x):
        conv_output = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_output)

