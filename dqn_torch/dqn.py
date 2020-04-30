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

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from tensorboardX import SummaryWriter

from prioritized_replay import PriorReplayBuffer

# 关于这个env的介绍，可以参考https://blog.csdn.net/cat_ziyan/article/details/101712107
# 对于原始图片，做了一部分裁剪，并设置channel=4（用于表现动作）
DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19.5

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 10**5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02

# 定义两个类，一个是经验池，一个是agent
class ExpBuf:
    def __init__(self, max_len):
        self.buffer = collections.deque(maxlen=max_len)

    def __len__(self):
        return len(self.buffer)

    def append(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        s, a, r, d, s_ = zip(*[self.buffer[idx] for idx in indices])
        return np.array(s), np.array(a), np.array(r), \
            np.array(d), np.array(s_)

class Agent:
    def __init__(self, env, buf, is_double):
        self.env = env
        self.buffer = buf
        self.double = is_double

    def step(self, net, state, epsilon=0.0, device="cpu"):
        if np.random.random() < epsilon:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                s = torch.tensor([state]).to(device)
                Q_val = net(s)
                a = int(Q_val.max(1)[1].item())
        s_, r, d, _  = self.env.step(a)
        # get all info and store them
        exp = [state, a, r, d, s_]
        self.buffer.append(exp)
        return s_, r, d

    def learn(self, eval_net, target_net, device="cpu"):
        s, a, r, d, s_ = self.buffer.sample(BATCH_SIZE)

        s = torch.tensor(s).to(device)
        a = torch.tensor(a).to(device)
        r = torch.tensor(r, dtype=torch.float).to(device)
        done_mask = torch.ByteTensor(d).to(device)
        s_ = torch.tensor(s_).to(device)

        Q_val = eval_net(s).gather(1, a.unsqueeze(-1)).squeeze(-1)
        if self.double:
            max_action_indices = eval_net(s_).max(1)[1]
            next_Q_val = target_net(s_).gather(1, max_action_indices.unsqueeze(-1)).squeeze(-1)
        else:
            next_Q_val = target_net(s_).max(1)[0]
        next_Q_val[done_mask] = 0.0
        expected_Q_val = r + GAMMA * next_Q_val.detach()
        
        return nn.MSELoss()(Q_val, expected_Q_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default="")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME)
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND)
    parser.add_argument("--model", default="dqn")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--prior", action="store_true", default=False)
    args = parser.parse_args()
    
    chosen_device = "cuda:%s" % (args.cuda)
    device = torch.device(chosen_device if args.cuda else "cpu")
    env = wrappers.make_env(args.env)
    
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        env.seed(args.seed)
    
    is_double = True if args.model == "double_dqn" else False
    is_dueling = True if args.model == "dueling_dqn" else False
    is_noisy = True if args.model == "noisy_net" else False

    if is_dueling:
        eval_net = base_model.DuelingDQN(env.observation_space.shape, env.action_space.n).to(device)
        target_net = base_model.DuelingDQN(env.observation_space.shape, env.action_space.n).to(device)
    elif is_noisy:
        eval_net = base_model.NoisyDQN(env.observation_space.shape, env.action_space.n).to(device)
        target_net = base_model.NoisyDQN(env.observation_space.shape, env.action_space.n).to(device)
    else:
        eval_net = base_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
        target_net = base_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    
    agent = Agent(env, ExpBuf(REPLAY_SIZE), is_double)
    epsilon = EPSILON_START
    optimizer = optim.Adam(eval_net.parameters(), lr=LEARNING_RATE)
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
            epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
            if is_noisy:
                epsilon = 0.0
            next_state, reward, done = agent.step(eval_net, state, epsilon, device)
            
            if frame_idx >= REPLAY_START_SIZE:
                # C轮强制替换
                if frame_idx % SYNC_TARGET_FRAMES == 0:
                    target_net.load_state_dict(eval_net.state_dict())
                optimizer.zero_grad()
                loss = agent.learn(eval_net, target_net, device)
                loss.backward()
                optimizer.step()
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
        writer.add_scalar("epsilon", epsilon, frame_idx)
        writer.add_scalar("speed", speed, frame_idx)
        writer.add_scalar("reward_100", mean_reward, frame_idx)
        writer.add_scalar("reward", reward, frame_idx)

        # for real_time 
        print("%d: done %d games, mean reward %.3f, acc reward %.3f, eps %.2f, speed %.2f f/s" % (
            frame_idx, len(total_rewards), mean_reward, acc_reward, epsilon, speed))
        if best_mean_reward is None or best_mean_reward < mean_reward:
            torch.save(eval_net.state_dict(), args.env + "-best.dat")
            if best_mean_reward is not None:
                print("best mean reward update %.3f -> %.3f" %(best_mean_reward, mean_reward))
            best_mean_reward = mean_reward
        if mean_reward > args.reward:
            print("solved in %d frames" % (frame_idx))
            break
    writer.close()
