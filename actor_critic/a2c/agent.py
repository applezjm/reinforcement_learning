#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim

class Agent():
    def __init__(self, net, value_coef, entropy_coef,
                 lr, optim_eps, max_grad_norm):
        self.net = net
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.optimizer = optim.RMSprop(self.net.parameters(), lr, eps=optim_eps,alpha=0.99)

    def step(self, states):
        with torch.no_grad():
            act_dist, vals = self.net(states)
            actions = act_dist.sample()
            action_log_probs = act_dist.log_probs(actions)
            return actions, action_log_probs, vals

    def learn(self, buf):
        self.optimizer.zero_grad()
        state_shape = buf.states.size()[2:]
        action_shape = buf.actions.size()[-1]

        act_dist, vals = self.net(buf.states[:-1].view(-1, *state_shape))
        # entropy loss
        loss_entropy = -act_dist.entropy().mean()
        
        num_steps, num_processes, _ = buf.rewards.size()
        values = vals.view(num_steps, num_processes, 1)
        action_log_probs = act_dist.log_probs(buf.actions.view(-1, action_shape)).view(num_steps, num_processes, 1)
        advantages = buf.returns[:-1] - values
        loss_value = advantages.pow(2).mean()
        loss_policy = -(advantages.detach() * action_log_probs).mean()
        # critic loss 
        #loss_value = nn.MSELoss()(buf.returns[:-1].view(-1, 1), vals)
        # actor loss
        #action_log_probs = act_dist.log_prob(buf.actions.view(-1)).unsqueeze(-1)
        #loss_policy = -(buf.advantages.view(-1, 1) * action_log_probs).mean()
        loss = loss_policy + loss_value * self.value_coef + loss_entropy * self.entropy_coef
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss_value.item(), loss_policy.item(), loss_entropy.item()
