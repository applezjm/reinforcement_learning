#!/usr/bin/env python
# -*- coding: utf-8 -*-

# based on NIPS 2013 DQN
# https://arxiv.org/abs/1312.5602
# details will be added soon

import tensorflow as tf
import numpy as np
import random
from collections import deque


class DQN:
    def __init__(self, state_dim, action_dim, learning_rate,
                 reward_decay, start_epsilon, end_epsilon,
                 batch_size, replay_size,):
        # state(状态)，action(动作)维度
        self.state_dim = state_dim
        self.action_dim = action_dim
        # 学习率
        self.learning_rate = learning_rate
        # 价值函数衰减率
        self.reward_decay = reward_decay
        # e_greedy的e
        self.epsilon = start_epsilon
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        # 经验重放的buffer大小
        self.replay_buffer = deque()
        self.replay_size = replay_size
        # minibatch的大小
        self.batch_size = batch_size

        self.cost = []

        self.build_net()
        # init session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build_net(self):
        # batch_input: s,r,a,s_
        self.state_input = tf.placeholder(tf.float32,
                                          [None, self.state_dim],
                                          name="state")
        self.action_input = tf.placeholder(tf.float32,
                                           [None, self.action_dim],
                                           name="action")
        self.q_value_input = tf.placeholder(tf.float32,
                                            [None, ],
                                            name="q_value")

        w_initializer = tf.random_normal_initializer(0.0, 0.3)
        b_initializer = tf.constant_initializer(0.1)

        with tf.variable_scope("q_network"):
            q1 = tf.layers.dense(self.state_input, 20, tf.nn.relu,
                                 kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name="q1")
            self.q_network = tf.layers.dense(q1, self.action_dim,
                                             kernel_initializer=w_initializer,
                                             bias_initializer=b_initializer,
                                             name="q")

        with tf.variable_scope("action_choose"):
            q_action = tf.reduce_sum(tf.multiply(self.q_network,
                                                 self.action_input),
                                     reduction_indices=1)
        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.square(self.q_value_input - q_action))
            self._train_op = tf.train.AdamOptimizer(
                self.learning_rate).minimize(self.loss)

    def store_transition(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > self.replay_size:
            self.replay_buffer.popleft()

    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        minibatch = random.sample(self.replay_buffer, self.batch_size)
        state_batch = [x[0] for x in minibatch]
        action_batch = [x[1] for x in minibatch] 
        reward_batch = [x[2] for x in minibatch] 
        next_state_batch = [x[3] for x in minibatch] 
        done_batch = [x[4] for x in minibatch] 

        q_value_batch = self.q_network.eval(session=self.sess,
            feed_dict={self.state_input: next_state_batch})
        expected_q_value_batch = reward_batch + \
            self.reward_decay * (1 - np.array(done_batch)) * \
            np.max(q_value_batch, axis=1)
        _, cost = self.sess.run([self._train_op, self.loss],
                                feed_dict={self.state_input:state_batch,
                                           self.action_input:action_batch,
                                           self.q_value_input:expected_q_value_batch})
        print(expected_q_value_batch)
        self.cost.append(cost)

    def e_greedy_action(self, state):
        q_value = self.q_network.eval(session=self.sess,
                                      feed_dict={self.state_input:[state]})[0]
        if random.random() <= self.epsilon:
            self.epsilon -= (self.start_epsilon - self.end_epsilon) / 10000
            return random.randint(0, self.action_dim - 1)
        else:
            self.epsilon -= (self.start_epsilon - self.end_epsilon) / 10000
            return np.argmax(q_value)

    def optimize_action(self, state):
        q_value = self.q_network.eval(session=self.sess,
                                      feed_dict={self.state_input:[state]})[0]
        return np.argmax(q_value)
