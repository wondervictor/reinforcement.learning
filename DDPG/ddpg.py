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

import gym
import random
import pickle
import numpy as np
import tensorflow as tf
from collections import namedtuple


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'terminate'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward, terminate):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(
            state=state,
            action=action,
            next_state=next_state,
            reward=reward,
            terminate=terminate
        )
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def save(self, path):
        f = open(path, 'wb')
        pickle.dump(self.memory, f)
        f.close()

    def __len__(self):
        return len(self.memory)


class Actor(object):

    def __init__(self, sess, num_states, num_actions, tau):
        self.sess = sess
        self.num_states = num_states
        self.num_actions = num_actions
        self.tau = tau

        # placeholders
        self.state = tf.placeholder(tf.float32, [None, num_states])
        self.q_grad = tf.placeholder(tf.float32, [None, num_actions])

        self.eval_scope = 'eval_actor_scope'
        self.target_scope = 'target_actor_scope'
        # build
        self.eval = self._build_eval()
        self.target = self._build_target()

        self._build_policy_gradient()
        self.optimizer = self._create_optimizer()
        self.train_op = self._train_op()


        self.target_update_op = self._update_target_op()

    def _build_target(self):
        eval_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_network')
        self.sess.run(tf.initialize_variables(eval_vars))
        with tf.variable_scope('target_actor_network'):
            output = self._build_network()
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_actor_network')
        target_init_list = [tf.assign(target_vars[k], eval_vars[k]) for k in range(len(eval_vars))]
        self.sess.run(target_init_list)
        return output

    def _build_eval(self):
        with tf.variable_scope('actor_network'):
            eval_net = self._build_network()
        return eval_net

    def _build_network(self):

        fc1 = tf.layers.dense(
            inputs=self.state,
            units=30,
            activation=tf.nn.leaky_relu,
            bias_initializer=tf.constant_initializer(0.1),
            name='fc1',
        )

        fc2 = tf.layers.dense(
            inputs=fc1,
            units=20,
            activation=tf.nn.leaky_relu,
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2',
        )

        output = tf.layers.dense(
            inputs=fc2,
            units=self.num_actions,
            bias_initializer=tf.constant_initializer(0.0),
            name='output',
        )

        return output

    def predict_action(self, state):
        action = self.sess.run(self.eval, feed_dict={self.state: state})
        return action

    def predict_target_action(self, state):
        action = self.sess.run(self.target, feed_dict={self.state: state})
        return action

    def _create_optimizer(self):
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        return optimizer

    def _build_policy_gradient(self):
        with tf.name_scope('loss'):
            eval_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_network')
            N = self.q_grad.shape[0]
            policy_gradient = tf.gradients(self.eval, eval_vars, grad_ys=self.q_grad) / N
        self.grad_and_vars = zip(eval_vars, policy_gradient)

    def _train_op(self):
        with tf.variable_scope('actor_train_op'):
            train_op = self.optimizer.apply_gradients(self.grad_and_vars, global_step=tf.contrib.frameworks.get_global_step())
            train_op_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_train_op')
            self.sess.run(tf.initialize_variables(train_op_vars))
        return train_op

    def _update_target_op(self):
        eval_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_network')
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_actor_network')

        update_op_list = [target_vars[i].assign((eval_vars[i]*self.tau + target_vars[i]*(1-self.tau)))
                          for i in range(len(eval_vars))]

        update_op = tf.group(*update_op_list)
        return update_op

    def update_eval(self, state, q_grad):
        feed_dict = {
            self.state: state,
            self.q_grad: q_grad
        }

        self.sess.run(self.train_op, feed_dict=feed_dict)

    def update_target(self):
        self.sess.run(self.target_update_op)


class Critic(object):

    # Q(s,a|theta)
    #
    #
    def __init__(self, sess, num_states, num_actions, tau):

        self.sess = sess
        self.num_actions = num_actions
        self.num_states = num_states
        self.tau = tau

        # placeholders
        self.state = tf.placeholder(tf.float32, [None, num_states])
        self.action = tf.placeholder(tf.float32, [None, num_actions])
        self.label = tf.placeholder(tf.float32, [None, 1])

        self.eval_scope = 'eval_critic_scope'
        self.target_scope = 'target_critic_scope'

        self.eval = self._build_eval()
        self.target = self._build_target()

        self.train_op = self._create_train_op()
        self.q_grad = self._get_q_grad_op()
        self.target_update_op = self._update_target_op()

    def _build_network(self):

        state_fc = tf.layers.dense(
            inputs=self.state,
            units=30,
            activation=tf.nn.leaky_relu,
            name='state_fc',
        )

        action_fc = tf.layers.dense(
            inputs=self.action,
            units=10,
            activation=tf.nn.leaky_relu,
            name='action_fc',
        )

        fc_concat = tf.concat([action_fc, state_fc], axis=1)

        fc = tf.layers.dense(
            inputs=fc_concat,
            units=30,
            activation=tf.nn.leaky_relu,
            name='fc',
        )

        output = tf.layers.dense(
            inputs=fc,
            units=1,
            name='output',
        )
        return output

    def predict_value(self, state, action):
        feed_dict = {
            self.state: state,
            self.action: action
        }

        val = self.sess.run(self.eval, feed_dict=feed_dict)
        return val

    def predict_target_value(self, state, action):
        feed_dict = {
            self.state: state,
            self.action: action
        }

        val = self.sess.run(self.target, feed_dict=feed_dict)
        return val

    def _build_target(self):
        with tf.variable_scope(self.target_scope):
            output = self._build_network()
        return output

    def _build_eval(self):
        with tf.variable_scope(self.eval_scope):
            output = self._build_network()
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.square(output-self.label))
        self.loss = loss
        return output

    def _get_q_grad_op(self):
        grad = tf.gradients(self.eval, self.action)
        return grad

    def get_q_gradients(self, state, action):
        feed_dict = {
            self.state: state,
            self.action: action
        }

        return self.sess.run(self.q_grad, feed_dict=feed_dict)

    def _update_target_op(self):
        eval_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.eval_scope)
        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.target_scope)

        update_op_list = [target_vars[i].assign((eval_vars[i]*self.tau + target_vars[i]*(1-self.tau)))
                          for i in range(len(eval_vars))]

        update_op = tf.group(*update_op_list)
        return update_op

    def _create_train_op(self):
        with tf.variable_scope('train_op'):
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(self.loss)
            train_op_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='train_op')
            self.sess.run(tf.variables_initializer(train_op_vars))
        return train_op

    def update_eval(self, state, action, label):
        feed_dict = {
            self.state: state,
            self.action: action,
            self.label: label
        }

        _loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)
        return _loss

    def update_target(self):
        self.sess.run(self.target_update_op)


def ddpg(env, num_states, num_actions, episode):

    actor = Actor(num_states, num_actions, 'eval')
    target_actor = Actor(num_states, num_actions, 'target')
    critic = Critic()


if __name__ == '__main__':

    env = gym.make('').unwrapped
