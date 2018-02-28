# -*- coding: utf-8 -*-

"""

MIT License

Copyright (c) 2018 Vic Chan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""


"""

Actor Critic Algorithm
"""

import gym
import numpy as np
import tensorflow as tf


class Actor(object):

    def __init__(self, state_space, action_space, ):

        self.state_space = state_space
        self.action_space = action_space

        self.state = tf.placeholder(tf.float32, shape=[None, state_space])
        self.action = tf.placeholder(tf.int32, shape=[None, action_space])
        self.delta = tf.placeholder(tf.float32, shape=[None, ])

        self._build_network()

    def _build_network(self):

        with tf.name_scope('actor_network'):

            fc1 = tf.layers.dense(
                inputs=self.state,
                units=32,
                activation=tf.nn.leaky_relu,
                bias_initializer=tf.constant_initializer(0.1),
                name='fc_1'
            )

            output = tf.layers.dense(
                inputs=fc1,
                units=self.action_space,
                activation=tf.nn.softmax,
                name='output'
            )
        self.output = output

        with tf.name_scope('actor_loss'):

            loss = tf.log(tf.reduce_sum(tf.multiply(output, self.action), reduction_indices=1))
            loss = tf.reduce_mean(tf.multiply(loss, self.delta))
        self.loss = loss

        with tf.name_scope('actor_train_op'):

            adam_optimizer = tf.train.AdamOptimizer(learning_rate=5e-4)
            train_op = adam_optimizer.minimize(loss)
        self.train_op = train_op

    def step(self, sess, state):
        prob = sess.run(self.output, feed_dict={self.state: [state]})
        action = np.random.choice(range(prob.shape[1]), p=prob.ravel())
        return action

    def update(self, sess, state, action, deltas):
        feed_dict = {
            self.state: state,
            self.action: action,
            self.delta: deltas
        }

        _loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)

        print("[Actor] Loss: {}".format(_loss))


class Critic(object):

    def __init__(self, state_space):
        self.state = tf.placeholder(tf.float32, [None, state_space])
        self.delta =  tf.placeholder(tf.float32, [None, ])

        self._build_network()

    def _build_network(self):

        with tf.name_scope('critic_network'):
            fc = tf.layers.dense(
                inputs=self.state,
                units=20,
                activation=tf.nn.leaky_relu,
                bias_initializer=tf.constant_initializer(0.1)
            )

            output = tf.layers.dense(
                inputs=fc,
                units=1,
                bias_initializer=tf.constant_initializer(0.1)
            )
        self.output = output

        with tf.name_scope('critic_loss'):
            loss = tf.reduce_mean(tf.square(output-self.delta))
        self.loss = loss

        with tf.name_scope('critic_train_op'):
            train_op = tf.train.AdamOptimizer(learning_rate=5e-4).minimize(loss)
        self.train_op = train_op

    def step(self, sess, state):
        value = sess.run(self.output, feed_dict={self.state: [state]})
        return value

    def update(self, sess, state, deltas):
        feed_dict = {
            self.state: state,
            self.delta: deltas
        }
        _loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        print("[Critic] Loss: {}".format(_loss))
