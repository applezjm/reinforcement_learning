#!/usr/bin/env python
# -*- coding: utf-8 -*-

# based on NIPS 2013 DQN
# https://arxiv.org/abs/1312.5602
# details will be added soon

import tensorflow as tf
import numpy as np
import random


class Actor(object):
    def __init__(self, sess, state_dim, action_dim):
        self.sess = sess

        # build net
        self.s = tf.placeholder(tf.float32, [1, state_dim], "state")
        self.a = tf.placeholder(tf.int32, None, "action")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")

        with tf.variable_scope("Actor"):
            a1 = tf.layers.dense(self.s, 20, tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0.0, 0.1),
                                 bias_initializer=tf.constant_initializer(0.1))
            self.act = tf.layers.dense(a1, action_dim, tf.nn.softmax,
                                 kernel_initializer=tf.random_normal_initializer(0.0, 0.1),
                                 bias_initializer=tf.constant_initializer(0.1))

        with tf.variable_scope("calc_entropy"):
            log_prob = tf.log(self.act[0, self.a])
            self.entropy = tf.reduce_mean(log_prob * self.td_error)

        with tf.variable_scope("train"):
            self.train_op = tf.train.AdamOptimizer(0.001).minimize(-self.entropy)

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        _, entropy = self.sess.run([self.train_op, self.entropy],
                                   feed_dict={self.s:s, self.a:a, self.td_error:td})
        return entropy

    def chose_action(self, s):
        s = s[np.newaxis, :]
        prob = self.sess.run(self.act,
                             feed_dict={self.s:s})
        return np.random.choice(np.arange(prob.shape[1]), p=prob.ravel())

class Critic(object):
    def __init__(self, sess, state_dim):
        self.sess = sess

        #build net
        self.s = tf.placeholder(tf.float32, [1, state_dim], "state")
        self.r = tf.placeholder(tf.float32, None, "reward")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "next_value")

        with tf.variable_scope("Critic"):
            c1 = tf.layers.dense(self.s, 20, tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0.0, 0.1),
                                 bias_initializer=tf.constant_initializer(0.1))
            self.v = tf.layers.dense(c1, 1, None,
                                     kernel_initializer=tf.random_normal_initializer(0.0, 0.1),
                                     bias_initializer=tf.constant_initializer(0.1))

        with tf.variable_scope("square_td_error"):
            # what is td_error?
            # r + gamma*v_ means the value fuction of (s,a)
            # v means the value fuction of (s)
            # the minus means the advantage fuction of (s,a) on the exact action
            self.td_error = self.r + 0.9 * self.v_ - self.v
            self.loss = tf.square(self.td_error)

        with tf.variable_scope("train"):
            self.train_op = tf.train.AdamOptimizer(0.01).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v_ = self.sess.run(self.v,
                           feed_dict={self.s:s_})
        td_error, a = self.sess.run([self.td_error, self.train_op],
                                    feed_dict={self.s:s,
                                               self.r:r,
                                               self.v_:v_})
        return td_error
