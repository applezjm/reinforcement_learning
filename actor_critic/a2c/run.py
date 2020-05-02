#!/usr/bin/env python
# -*- coding: utf-8 -*-

import base_model
import wrappers

import os
import argparse
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

from buf import ExperienceBuffer

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 18

GAMMA = 0.95
BATCH_SIZE = 32
REPLAY_SIZE = 32
LEARNING_RATE = 1e-3
VAL_BETA = 1.0
ENTROPY_BETA = 0.002
CLIP_GRAD = 0.1


class Agent:
    def __init__(self, env, buf):
        self.env = env
        self.buffer = buf

    def step(self, net, state, device="cpu"):
        with torch.no_grad():
            s = np.array([state])
            s = torch.tensor(s).to(device)
            a = net(s)[0].multinomial(1, replacement=False).item()
        s_, r, d, _  = self.env.step(a)
        # get all info and store them
        exp = [state, a, r, d, s_]
        self.buffer.store(exp)
        return s_, r, d

    def learn(self, net, optimizer, device="cpu"):
        optimizer.zero_grad()

        s, a, r, d, s_ = self.buffer.sample(BATCH_SIZE)

        s = torch.tensor(s).to(device)
        a = torch.tensor(a).to(device)
        r = torch.tensor(r, dtype=torch.float).to(device)
        done_mask = torch.ByteTensor(d).to(device)
        s_ = torch.tensor(s_).to(device)

        # loss_policy + loss_value + loss_entropy
        prob, val = net(s)
        _, next_val = net(s_)

        val, next_val = val.squeeze(-1), next_val.squeeze(-1)
        target_val = r + GAMMA * next_val.detach()
        target_val[done_mask] = 0.0
        loss_value = VAL_BETA * F.mse_loss(val, target_val)

        adv_val = target_val - val.detach()
        log_prob_adv = -adv_val * prob.gather(1, a.unsqueeze(-1)).squeeze(-1).log()
        # policy gradient(actually ascent)
        loss_policy = log_prob_adv.mean()

        # 增加entropy，利于exploration
        loss_entropy = ENTROPY_BETA * (prob * prob.log()).sum(dim=1).mean()
        
        loss = loss_entropy + loss_value + loss_policy
        loss.backward()
        nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
        optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default="")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME)
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND)
    parser.add_argument("--model", default="a2c")
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()
    
    chosen_device = "cuda:%s" % (args.cuda)
    device = torch.device(chosen_device if args.cuda else "cpu")
    env = wrappers.make_env(args.env)
    
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        env.seed(args.seed)
    
    net = base_model.A2C(env.observation_space.shape, env.action_space.n).to(device)
    
    buf = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buf)
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)
    writer = SummaryWriter(comment="-" + args.env + "-" + args.model)

    total_rewards = []
    acc_reward = None
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None

    while True:
        state = env.reset()
        reward_epsiode = 0.0
        is_done = False
        while not is_done:
            frame_idx += 1
            next_state, reward, done = agent.step(net, state, device)
            
            if len(buf) >= BATCH_SIZE:
                agent.learn(net, optimizer, device)
                buf.clear()
            is_done = done
            state = next_state
            reward_epsiode += reward

        total_rewards.append(reward_epsiode)
        speed = (frame_idx - ts_frame) / (time.time() - ts)
        ts_frame = frame_idx
        ts = time.time()
        mean_reward = np.mean(total_rewards[-100:])
        if acc_reward:
            acc_reward = 0.1 * reward_epsiode + 0.9 * acc_reward
        else:
            acc_reward = reward_epsiode

        # save data for final pic
        writer.add_scalar("speed", speed, frame_idx)
        writer.add_scalar("reward_100", mean_reward, frame_idx)
        writer.add_scalar("reward", reward, frame_idx)

        # for real_time 
        print("%d: done %d games, mean reward %.3f, acc reward %.3f, speed %.2f f/s" % (
            frame_idx, len(total_rewards), mean_reward, acc_reward, speed))
        if best_mean_reward is None or best_mean_reward < mean_reward:
            torch.save(net.state_dict(), args.env + "-best.dat")
            if best_mean_reward is not None:
                print("best mean reward update %.3f -> %.3f" %(best_mean_reward, mean_reward))
            best_mean_reward = mean_reward
        if mean_reward > args.reward:
            print("solved in %d frames" % (frame_idx))
            break
    writer.close()
