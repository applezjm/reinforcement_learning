#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch


class Buffer(object):
    def __init__(self, num_steps, num_processes, state_shape, action_shape, device):
        self.states = torch.zeros(num_steps+1, num_processes, *state_shape).to(device)
        self.actions = torch.zeros(num_steps, num_processes, 1).to(device)
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1).to(device)
        self.rewards = torch.zeros(num_steps, num_processes, 1).to(device)
        self.masks = torch.zeros(num_steps, num_processes, 1).to(device) 
        self.values = torch.zeros(num_steps+1, num_processes, 1).to(device)

        self.returns = torch.zeros(num_steps+1, num_processes, 1).to(device)
        self.advantages = torch.zeros(num_steps, num_processes, 1).to(device)

        self.num_steps = num_steps
        self.step = 0

    def insert(self, states, actions, rewards, masks, action_log_probs, values):
        self.states[self.step+1].copy_(states)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step].copy_(masks)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.values[self.step].copy_(values)

        self.step = (self.step + 1) % self.num_steps

    def reset(self):
        self.states[0].copy_(self.states[-1])

    def calc_returns(self, use_gae, gae_lambda, gamma):
        if use_gae:
            gae = 0.0
            for step in reversed(range(self.rewards.size(0))):
                one_step_delta = self.rewards[step] + gamma * \
                    self.values[step+1] * self.masks[step] - self.values[step]
                gae = one_step_delta + gamma * gae_lambda * self.masks[step] * gae
                self.advantages[step] = gae
                self.returns[step] = gae + self.values[step]
        else:
            self.returns[-1] = self.values[-1]
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step+1] * gamma * \
                    self.masks[step] + self.rewards[step]
                self.advantages[step] = self.returns[step] - self.values[step]





