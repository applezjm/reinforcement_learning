#!/usr/bin/env python
# -*- coding: utf-8 -*-

# based on NIPS 2013 DQN
# https://arxiv.org/abs/1312.5602
# details will be added soon

import tensorflow as tf
import numpy as np
import random
from collections import deque


class DoubleDQN:
    def __init__(self, state_dim, action_dim, learning_rate,
                 reward_decay, start_epsilon, end_epsilon,
                 batch_size, replay_size, replace_iter):
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

        # C:强制eval替换target
        self.replace_iter = replace_iter
        self.learn_step_counter = 0

        self.cost = []

        self.build_net()

        # copy params
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="target_net")
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="eval_net")

        with tf.variable_scope("params_replacement"):
            self.replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        # init session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build_net(self):
        # batch_input: s,r,a,s_
        self.state_input = tf.placeholder(tf.float32,
                                          [None, self.state_dim],
                                          name="state")
        self.next_state_input = tf.placeholder(tf.float32,
                                               [None, self.state_dim],
                                               name="next_state")
        self.action_input = tf.placeholder(tf.float32,
                                           [None, self.action_dim],
                                           name="action")
        self.reward_input = tf.placeholder(tf.float32,
                                           [None, ],
                                           name="reward")
        self.chosen_action_input = tf.placeholder(tf.float32,
                                                  [None, self.action_dim],
                                                  name="chosen_action")

        w_initializer = tf.random_normal_initializer(0.0, 0.3)
        b_initializer = tf.constant_initializer(0.1)

        with tf.variable_scope("eval_net"):
            e1 = tf.layers.dense(self.state_input, 20, tf.nn.relu,
                                 kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name="e1")
            self.eval_net = tf.layers.dense(e1, self.action_dim,
                                            kernel_initializer=w_initializer,
                                            bias_initializer=b_initializer, name="eval")

        with tf.variable_scope("target_net"):
            t1 = tf.layers.dense(self.next_state_input, 20, tf.nn.relu,
                                 kernel_initializer=w_initializer,
                                 bias_initializer=b_initializer, name="t1")
            self.target_net = tf.layers.dense(t1, self.action_dim,
                                              kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer, name="target")

        with tf.variable_scope("q_target"):
            max_action_pos_ = tf.argmax(self.chosen_action_input, axis=1)
            max_action_pos = tf.cast(max_action_pos_, dtype=tf.int32)
            a_indices = tf.stack([tf.range(tf.shape(max_action_pos)[0], dtype=tf.int32), max_action_pos], axis=1) 
            q_target = tf.gather_nd(params=self.target_net, indices=a_indices)
            q_real_target = self.reward_input + self.reward_decay * q_target
            self.q_target = tf.stop_gradient(q_real_target)
        
        with tf.variable_scope("q_eval"):
            # also valid for stack + gather_nd
            self.q_eval = tf.reduce_sum(tf.multiply(self.eval_net, self.action_input),
                                        reduction_indices=1, name="q_eval")

        with tf.variable_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.square(self.q_target - self.q_eval))
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
        if self.learn_step_counter % self.replace_iter == 0:
            print("params replacement")
            self.sess.run(self.replace_op)

        minibatch = random.sample(self.replay_buffer, self.batch_size)
        state_batch = [x[0] for x in minibatch]
        action_batch = [x[1] for x in minibatch] 
        reward_batch = [x[2] for x in minibatch] 
        next_state_batch = [x[3] for x in minibatch] 
        done_batch = [x[4] for x in minibatch] 
        chosen_action = self.sess.run(self.eval_net,
                                      feed_dict={self.state_input:next_state_batch})
        _, cost = self.sess.run([self._train_op, self.loss],
                                feed_dict={self.state_input:state_batch,
                                           self.next_state_input:next_state_batch,
                                           self.action_input:action_batch,
                                           self.reward_input:reward_batch,
                                           self.chosen_action_input:chosen_action})
        self.cost.append(cost)
        self.learn_step_counter += 1

    def e_greedy_action(self, state):
        q_value = self.eval_net.eval(session=self.sess,
                                    feed_dict={self.state_input:[state]})[0]
        if random.random() <= self.epsilon:
            self.epsilon -= (self.start_epsilon - self.end_epsilon) / 10000
            return random.randint(0, self.action_dim - 1)
        else:
            self.epsilon -= (self.start_epsilon - self.end_epsilon) / 10000
            return np.argmax(q_value)

    def optimize_action(self, state):
        q_value = self.eval_net.eval(session=self.sess,
                                    feed_dict={self.state_input:[state]})[0]
        return np.argmax(q_value)
