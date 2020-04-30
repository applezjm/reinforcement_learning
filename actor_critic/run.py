#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gym
import tensorflow as tf
from actor_critic_origin import Actor, Critic
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

ENV_NAME = "CartPole-v0"
EPISODE = 2000
# 在200个reward获取后，游戏结束
STEP = 300

def run():
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    sess = tf.Session()
    actor = Actor(sess, env.observation_space.shape[0], env.action_space.n)
    critic = Critic(sess, env.observation_space.shape[0])
    sess.run(tf.global_variables_initializer())

    for episode in range(EPISODE):
        # 初始state
        state = env.reset()
        # play and train method
        total_reward = 0
        for step in range(STEP):
            action = actor.chose_action(state)
            next_state, reward, done, _ = env.step(action)
            # 定义reward？
            if done:
                reward = -20
            total_reward += reward
            
            td_error = critic.learn(state, reward, next_state)
            actor.learn(state, action, td_error)
            state = next_state
            if done:
                print("epsiode:{}, reward:{}".format(episode, total_reward))
                break
    
if __name__ == "__main__":
    run()

