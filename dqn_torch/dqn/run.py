#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
from dqn import DQN
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

ENV_NAME = "CartPole-v0"
EPISODE = 300
# 在200个reward获取后，游戏结束
STEP = 300
TEST_EPSIODE = 10

def run():
    env = gym.make(ENV_NAME)
    agent = DQN(env.observation_space.shape[0],
                env.action_space.n,
                0.001, 0.9, 0.5, 0.01, 32, 10000, 300)

    for episode in range(EPISODE):
        # 初始state
        state = env.reset()
        # play and train method
        for step in range(STEP):
            action = agent.e_greedy_action(state)
            next_state, reward, done, _ = env.step(action)
            # 定义reward？
            reward = -1 if done else 0.1
            
            agent.store_transition(state, action, reward, next_state, done)

            agent.learn()
            
            state = next_state
            if done:
                break

        # test every 100 epsiode
        # look for more ways to define loss
        if episode % 10 == 0:
            total_reward = 0
            for _ in range(TEST_EPSIODE):
                state = env.reset()
                for step in range(STEP):
                    action = agent.optimize_action(state)
                    state, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
            average_reward = total_reward / TEST_EPSIODE
            print("epsiode:{}, ave_reward:{}".format(episode, average_reward))
    
    # draw loss pic
    accumulate = np.add.accumulate(agent.cost)
    plt.plot(np.arange(len(accumulate)) + 1, accumulate / (np.arange(len(accumulate)) + 1))
    plt.ylabel("cost")
    plt.xlabel("training steps")
    plt.savefig("cost.png")

if __name__ == "__main__":
    run()

