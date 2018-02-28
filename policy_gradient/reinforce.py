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
Policy Gradient: REINFORCE Algorithm

For each step of the episode t = 0, . . . , T − 1:
    Gt ← return from step t
    δ ← G_t − vˆ(S_t, w)
    w ← w + βδ∇_wvˆ(S_t , w)
    θ ← θ + αγtδ∇_θlogπ(At|St, θ)
"""

import gym
import pickle
import numpy as np
import tensorflow as tf
from itertools import count
from collections import namedtuple


class Policy(object):

    def __init__(self, num_state, num_actions):
        self.num_state = num_state
        self.num_actions = num_actions

        self.states = tf.placeholder(tf.float32, shape=[None, num_state])
        self.actions = tf.placeholder(tf.int32, shape=[None, num_actions])
        self.value = tf.placeholder(tf.float32, shape=[None, ])

        # build network
        self._build_network()

    def _build_network(self):
        with tf.name_scope('policy_network'):
            fc1 = tf.layers.dense(self.states, units=self.num_state * 5, activation=tf.nn.tanh, name='fc1')
            output = tf.layers.dense(fc1, units=self.num_actions, name='output')
        self.output_layer = tf.nn.softmax(output)

        with tf.name_scope('loss'):
            action_probs = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=self.actions)
            loss = tf.reduce_mean(action_probs * self.value)
        self.loss = loss

        with tf.name_scope('train_op'):
            adam_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            train_op = adam_optimizer.minimize(loss)
        self.train_op = train_op

    def step(self, sess, state):
        # predict actions from pi(a|s)
        prob = sess.run(self.output_layer, feed_dict={self.states: [state]})
        action = np.random.choice(range(prob.shape[1]), p=prob.ravel())
        return action

    def update(self, sess, state, actions, value):
        # update policy by policy gradient

        feed_dict = {
            self.states: state,
            self.actions: actions,
            self.value: value
        }
        _loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        print("[Policy Network] Loss: {}".format(_loss))


class Value(object):

    def __init__(self, num_state):
        self.num_state = num_state
        self.state = tf.placeholder(tf.float32, shape=[None, num_state])
        self.value = tf.placeholder(tf.float32, shape=[None, ])

        self._build_network()

    def _build_network(self):
        with tf.name_scope('value_network'):
            output = tf.layers.dense(
                inputs=self.state,
                units=1,
                name='fc',
                activation=tf.nn.leaky_relu
            )
        self.output = output

        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.square(output - self.value))

        self.loss = loss

        with tf.name_scope('train_op'):
            adam_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            train_op = adam_optimizer.minimize(loss)
        self.train_op = train_op

    def step(self, sess, state):
        out = sess.run(self.output, feed_dict={self.state: [state]})
        return out

    def update(self, sess, state, value):

        feed_dict = {
            self.state: state,
            self.value: value
        }

        _loss, _ = sess.run([self.loss, self.train_op], feed_dict=feed_dict)

        print("[Value Network] Loss: {}".format(_loss))


def reinforce(env, gamma, num_actions, num_state, epoches):

    policy_net = Policy(num_state=num_state, num_actions=num_actions)
    value_net = Value(num_state=num_state)

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        overall_rewards = []
        overall_step = []

        for epoch in xrange(epoches):
            episode_state = []
            episode_reward = []
            episode_action = []
            state = env.reset()
            overall_rewards.append(0.)
            overall_step.append(0)
            # env.render()
            for t in count():

                action = policy_net.step(sess, state)
                next_state, reward, done, _ = env.step(action)

                episode_state.append(state)
                episode_reward.append(reward)
                zero_action = [0, 0, 0]
                zero_action[action] = 1
                episode_action.append(zero_action)

                overall_rewards[epoch] += reward
                overall_step[epoch] = t
                if done:
                    break
                state = next_state

            print("Episode: {} Steps: {} Score: {}".format(epoch, overall_step[epoch], overall_rewards[epoch]))
            # learn
            advantage = []
            total_return = []
            for t in range(len(episode_reward)):
                total_value = sum(gamma ** i * m for i, m in enumerate(episode_reward[t:]))
                baseline_value = value_net.step(sess, episode_state[t])
                _advantage = total_value - baseline_value[0][0]

                total_return.append(total_value)
                advantage.append(_advantage)

            value_net.update(
                sess=sess,
                state=episode_state,
                value=total_return
            )

            policy_net.update(
                sess=sess,
                state=episode_state,
                value=advantage,
                actions=episode_action
            )


if __name__ == '__main__':

    env = gym.make('MountainCar-v0').unwrapped
    # print(env.action_space, env.observation_space)
    # reinforce(env, 0.90)
    action_space = 3
    obs_space = 2

    reinforce(env, gamma=0.95, num_state=obs_space, num_actions=action_space, epoches=100)



