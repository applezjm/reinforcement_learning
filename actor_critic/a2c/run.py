#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, "../")
from envs import make_vec_envs 
from storage import Buffer
from model import AC
from argument import get_args
from agent import Agent

import os
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils as nn_utils

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from tensorboardX import SummaryWriter

if __name__ == "__main__":
    args = get_args()
    
    device = torch.device(args.cuda)
    torch.set_num_threads(1)
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)  
    
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    net = AC(envs.observation_space.shape,
                         envs.action_space.n, args.batchnorm).to(device) 
    buf = Buffer(args.num_steps, args.num_processes,
                 envs.observation_space.shape, envs.action_space, device)
    agent = Agent(net, args.value_coef, args.entropy_coef, args.lr, args.optim_eps,
                  args.max_grad_norm)
    
    writer = SummaryWriter(comment="-" + args.name)

    mean_50_rewards = collections.deque(maxlen=50)
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    mean_reward = 0.0

    # initial environment
    states = envs.reset()
    buf.states[0].copy_(states)
    while mean_reward < args.reward_bound:
        for step in range(args.num_steps):
            # agent负责动作选择 
            actions, action_log_probs, vals = agent.step(states)
            # env负责与环境交互
            states, rewards, dones, infos = envs.step(actions)
            # 抽取reward，便于分析
            for info in infos:
                if "episode" in info.keys():
                    mean_50_rewards.append(info["episode"]['r'])
            masks = torch.Tensor(1 - np.array(dones)).unsqueeze(-1)
            # buffer负责存储
            buf.insert(states, actions, rewards, masks, action_log_probs, vals)    
        _, _, last_vals =  agent.step(states)
        buf.values[-1].copy_(last_vals)
        # 计算当步return+adv 
        buf.calc_returns(args.gae, args.gae_lambda, args.gamma)
        # update network
        loss_value, loss_policy, loss_entropy = agent.learn(buf)
        buf.reset()

        frame_idx += args.num_steps * args.num_processes
        speed = (frame_idx - ts_frame) / (time.time() - ts)
        ts_frame = frame_idx
        ts = time.time()
       
        if len(mean_50_rewards) > 0:
            mean_reward = np.mean(mean_50_rewards)

        writer.add_scalar("speed", speed, frame_idx)
        writer.add_scalar("reward_50", mean_reward, frame_idx)
        writer.add_scalar("entropy loss", loss_entropy, frame_idx)
        writer.add_scalar("value loss", loss_value, frame_idx)
        writer.add_scalar("policy gradient loss", loss_policy, frame_idx)

        # for real_time 
        print("done %d games, mean reward %.3f, speed %.2f f/s" % (
            frame_idx, mean_reward, speed))
    torch.save(net.state_dict(), args.env_name + args.name + "-best.dat")
    writer.close()
