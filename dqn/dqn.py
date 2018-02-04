# -*- coding: utf-8 -*-
import os
import gym
import math
import time
import torch
import random
import pickle
import numpy as np
import torch.nn as nn
from itertools import count
import torch.optim as optimizer
import matplotlib.pyplot as plt
from collections import namedtuple
from torch.autograd import Variable


# Environment
env = gym.make('MountainCar-v0').unwrapped


def get_screen():
    screen = env.render(mode='rgb_array')
    plt.show(screen)


for i in xrange(10):
    env.reset()
    get_screen()
    for t in count():
        q_value = np.random.rand(3)
        action_index = np.argmax(q_value)
        q_value = q_value[action_index]
        print("q_value:%s" % q_value)
        _, reward, done, _ = env.step(action_index)
        get_screen()

        if done:
            break


class QNet(nn.Module):

    def __init__(self, input_dim, output_actions, use_gpu=False):
        super(QNet, self).__init__()
        self.use_gpu = use_gpu
        self.input_dim = input_dim
        self.output_actions = output_actions

        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, output_actions*3),
        )

    def forward(self, x):

        x = self.fc_layers(x)
        x = x.view(-1, self.output_actions, 3)
        return x


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


class DQN(object):

    def __init__(self, config):
        """
        :param config: 模型配置参数
        """
        self.num_actions = config['num_actions']
        self.input_dim = config['input_dim']
        self.use_gpu = config['use_gpu']

        self.epsilon_start = config['epsilon_start']
        self.epsilon_end = config['epsilon_end']
        self.epsilon_decay = config['epsilon_decay']

        self.gamma = config['gamma']

        self.update_target = config['update_target']
        self.epoches = config['epoches']
        self.train_batch_size = config['train_batch_size']

        # 全局计时器
        self.step_counter = 0

        self.eval_net = QNet(self.input_dim, self.num_actions, self.use_gpu)
        self.target_net = QNet(self.input_dim, self.num_actions, self.use_gpu)

        learning_rate = config['learning_rate']

        self.optimizer = optimizer.Adam(params=self.eval_net.parameters(), lr=learning_rate)
        self.loss_criterion = nn.MSELoss()

        self.replay_memory_size = config['replay_memory_size']
        self.replay_memory = ReplayMemory(self.replay_memory_size)

        self.current_state = None

    def convert_state_to_variable(self, state):
        """
        将内部metric转为torch variable
        :param state:
        :return:
        """
        return Variable(torch.FloatTensor(state))

    def choose_action(self, step, current_state):
        """
        选择action
        :param step: 步骤index
        :param current_state: 当前metric状态信息
        :return:
        """
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start-self.epsilon_end)*math.exp(-step*self.epsilon_decay)
        if sample < eps_threshold:
            q_value = np.random.rand(2)
            action_index = np.argmax(q_value)
            q_value = q_value[action_index]
            print("step: %s choose from random" % step)
        else:
            print("step: %s choose from net" % step)
            state = self.convert_state_to_variable(current_state)
            if len(state.size()) < 2:
                state = state.unsqueeze(0)
            q_value = self.eval_net(state).squeeze(0).data.numpy()
            action_index = np.argmax(q_value)
            q_value = q_value[action_index]

        return action_index, q_value

    def apply_action(self, env, actions):
        reward, next_state, terminate = env.apply_knobs(actions)
        return reward, next_state, terminate

    def store_memory(self, state, action, reward, next_state, terminate):
        # 存储历史
        self.replay_memory.push(
            state=state,
            next_state=next_state,
            reward=reward,
            action=action,
            terminate=terminate
        )

    def save_parameters(self, param_dir, step_counter):
        """
        保存模型参数
        :param param_dir:
        :param step_counter:
        :return:
        """
        if not os.path.exists(param_dir):
            os.mkdir(param_dir)
        torch.save(self.eval_net.state_dict(), param_dir+'eval_net_%s_param.pth' % step_counter)
        torch.save(self.target_net.state_dict(), param_dir+'target_net_%s_param.pth' % step_counter)

    def save_to_file(self, path):
        self.replay_memory.save(path)

    def get_batch_data(self):
        """
        返回batch data
        :return:
        """

        batch_data = self.replay_memory.sample(batch_size=self.train_batch_size)
        return batch_data

    def learn(self, episode):
        # 学习
        if self.step_counter % self.update_target == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        batch_data = self.get_batch_data()
        batch_state = [x.state for x in batch_data]
        batch_action = [x.action for x in batch_data]
        batch_reward = [x.reward for x in batch_data]
        batch_next_state = [x.next_state for x in batch_data if x.next_state is not None]
        batch_terminate = [x.terminate for x in batch_data]
        batch_state = self.convert_state_to_variable(batch_state)
        batch_next_state = self.convert_state_to_variable(batch_next_state)
        batch_reward = Variable(torch.FloatTensor(batch_reward))
        batch_action = Variable(torch.LongTensor(batch_action))
        q_eval = self.eval_net(batch_state)
        _, _, q_eval_value = self.action_selector.forward(q_eval, predict=False, action=batch_action)

        q_next = self.target_net(batch_next_state).detach()
        _, _, q_next_value = self.action_selector.forward(q_next)
        q_target = Variable(torch.zeros(self.train_batch_size))
        for i in xrange(self.train_batch_size):
            if batch_terminate[i]:
                q_target[i] = batch_reward[i]
            else:
                q_target[i] = batch_reward[i] + self.gamma * q_next_value[i]
        loss = self.loss_criterion(q_eval_value, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        log_info = "[Episode:{}][Step:{}]Loss:{}".format(episode, self.step_counter, loss.data[0])
        print(log_info)
        with open('training.log', 'a+') as f:
            f.write(log_info + '\n')
        if (self.step_counter+1) % 500 == 0:
            self.save_parameters('params/', self.step_counter)

    def train(self, env, provided_memory):
        """
        :param provided_memory: 提供部分训练数据
        :return:
        """
        self.current_state = env.get_initial_state()
        if len(provided_memory) > 0:
            for i in xrange(len(provided_memory)):
                self.store_memory(
                    provided_memory[i][0],
                    provided_memory[i][1],
                    provided_memory[i][2],
                    provided_memory[i][3],
                    provided_memory[i][4]
                )
        episode_durations = []
        for episode in xrange(self.epoches):
            self.current_state = env.get_initial_state()
            for t in count():
                state = self.current_state
                action, q_value = self.choose_action(self.step_counter, current_state=state)
                reward, state_, terminate = self.apply_action(env, action)

                print("Step: %s" % self.step_counter)
                print("Q_value=%s" % q_value)
                print("Reward: %s" % reward)
                print("Terminate: %s" % terminate)

                next_state = state_

                self.store_memory(
                    state=state,
                    action=action,
                    next_state=next_state,
                    reward=reward,
                    terminate=terminate
                )

                self.current_state = next_state
                if len(self.replay_memory) > 5 * self.train_batch_size:
                    self.learn(episode)

                if terminate:
                    episode_durations.append(t+1)
                    break

                self.step_counter += 1

            self.save_to_file('memory/memory_%s.pkl' % int(time.time()))









