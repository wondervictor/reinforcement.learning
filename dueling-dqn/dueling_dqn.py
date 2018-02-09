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
import os
import cv2
import math
import time
import torch
import random
import pickle
import numpy as np
import torch.nn as nn
from itertools import count
import matplotlib.pyplot as plt
import torch.optim as optimizer
import torch.nn.functional as F
from collections import namedtuple
from torch.autograd import Variable
from flappy_bird import flappy_bird


class ValueNet(nn.Module):

    def __init__(self):
        super(ValueNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(9 * 16 * 64, 256),
            nn.LeakyReLU(negative_slope=0.03),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.03),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.fc(x)


class AdvantageNet(nn.Module):

    def __init__(self, output_actions):
        super(AdvantageNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(9 * 16 * 64, 512),
            nn.LeakyReLU(negative_slope=0.03),
            nn.Linear(512, 64),
            nn.LeakyReLU(negative_slope=0.03),
            nn.Linear(64, 2),
            nn.Softmax()
        )

    def forward(self, x):
        return self.fc(x)


class DuelingNet(nn.Module):

    def __init__(self, in_chans, output_actions):
        super(DuelingNet, self).__init__()

        self.in_conv1 = nn.Conv2d(in_channels=in_chans, out_channels=8, kernel_size=5, padding=2)
        self.in_conv2 = nn.Conv2d(in_channels=in_chans, out_channels=8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.03),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=64),
        )

        self.advantage_net = AdvantageNet(output_actions)
        self.value_net = ValueNet()

    def forward(self, x):
        x1 = self.pool(F.leaky_relu(self.in_conv1(x), negative_slope=0.3))
        x2 = self.pool(F.leaky_relu(self.in_conv2(x), negative_slope=0.3))

        x = torch.cat([x1, x2], dim=1)
        x = self.convs(x)
        x = x.view(-1, 9 * 16 * 64)
        value = self.value_net(x)
        advatage = self.advantage_net(x)
        mean_advantage = torch.mean(advatage, 1).unsqueeze(1)
        out = value + (advatage - mean_advantage)
        return out


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


class DuelingDQN(object):

    def __init__(self, config, env):

        self.num_actions = config['num_actions']
        self.use_gpu = config['use_gpu']

        self.epsilon_start = config['epsilon_start']
        self.epsilon_end = config['epsilon_end']
        self.epsilon_decay = config['epsilon_decay']

        self.gamma = config['gamma']

        self.update_target = config['update_target']
        self.epoches = config['epoches']
        self.batch_size = config['batch_size']

        learning_rate = config['learning_rate']

        self.replay_memory_size = config['replay_memory_size']

        self.env = env

        self.target_net = DuelingNet(1, self.num_actions)
        self.eval_net = DuelingNet(1, self.num_actions)

        self.loss_criterion = nn.MSELoss()
        # optimizer
        self.optimizer = optimizer.Adam(lr=learning_rate, params=self.eval_net.parameters())
        # replay memory
        self.replay_memory = ReplayMemory(self.replay_memory_size)

        self.step_counter = 0

    def choose_action(self, step, state, random_action=False):
        sample = random.random()
        eps_threshold = self.epsilon_end + (self.epsilon_start-self.epsilon_end)*math.exp(-step*self.epsilon_decay)

        action = [0, 0]
        if random_action:
            q_value = np.random.random()
            if q_value >= 0.5:
                action_index = 1
            else:
                q_value = 1 - q_value
                action_index = 0
            print("step: %s choose from random action: %s" % (step, action_index))
            action[action_index] = 1

            return action, q_value

        if sample < eps_threshold:
            q_value = np.random.random()
            if q_value >= 0.5:
                action_index = 1
            else:
                q_value = 1 - q_value
                action_index = 0
            print("step: %s choose from random action: %s" % (step, action_index))
        else:
            state = Variable(torch.FloatTensor(state)).unsqueeze(0)
            q_value = self.eval_net(state).squeeze(0).data.numpy()
            action_index = np.argmax(q_value)
            q_value = q_value[action_index]
            print("step: %s choose from net action: %s" % (step, action_index))

        action[action_index] = 1
        return action, q_value

    def apply_action(self, action):
        state, reward, terminate = env.frame_step(action)
        state = self.get_state(state)
        return state, reward, terminate

    def get_state(self, x):
        img = np.transpose(x, (1, 0, 2))
        img = cv2.resize(img, (144, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.array(img)
        img = img / 255.0
        img = np.reshape(img, (1, 144, 256))

        return np.array(img)

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
        batch_data = self.replay_memory.sample(batch_size=self.batch_size)
        return batch_data

    def learn(self, episode):
        if self.step_counter % self.update_target == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        batch_data = self.get_batch_data()
        batch_state = [x.state for x in batch_data]
        batch_action = [np.argmax(x.action) for x in batch_data]
        batch_reward = [x.reward for x in batch_data]
        batch_next_state = [x.next_state for x in batch_data]
        batch_terminate = [x.terminate for x in batch_data]
        batch_state = Variable(torch.FloatTensor(batch_state))
        batch_next_state = Variable(torch.FloatTensor(batch_next_state))
        batch_reward = Variable(torch.FloatTensor(batch_reward))

        q_eval = self.eval_net(batch_state)
        q_eval_value = Variable(torch.zeros(self.batch_size))
        for i in xrange(self.batch_size):
            q_eval_value[i] = q_eval[i, batch_action[i]]

        q_next = self.eval_net(batch_next_state).detach()
        _, action_indexes = torch.max(q_next, 1)
        q_target = Variable(torch.zeros(self.batch_size))
        q_next = self.target_net(batch_next_state).detach()
        q_eval_value = Variable(torch.zeros(self.batch_size))
        action_indexes = action_indexes.data.numpy()
        for i in xrange(self.batch_size):
            q_eval_value[i] = q_eval[i, batch_action[i]]
            q_target[i] = q_next[i, action_indexes[i]]

        for i in xrange(self.batch_size):
            if batch_terminate[i]:
                q_target[i] = batch_reward[i]
            else:
                q_target[i] = batch_reward[i] + self.gamma * q_target[i]

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

    def train(self):

        episode_durations = []
        for episode in xrange(self.epoches):
            state, _, _ = self.apply_action([1, 0])

            for t in count():
                if len(self.replay_memory) < 100 * self.batch_size:
                    random_action = True
                else:
                    random_action = False

                action, q_value = self.choose_action(self.step_counter, state, random_action)
                state_, reward, terminate = self.apply_action(action)

                print("[Step]:{} [Q_Value]: {} [Reward]: {} [Terminate]: {}"
                      .format(self.step_counter, q_value, reward, terminate))

                next_state = state_

                self.store_memory(
                    state=state,
                    action=action,
                    next_state=next_state,
                    reward=reward,
                    terminate=terminate
                )
                if not random_action:
                    self.learn(episode)
                self.step_counter += 1
                if terminate:
                    episode_durations.append(t+1)
                    break
            if (self.step_counter+1) % 3000 == 0:
                self.save_to_file('memory/memory_%s.pkl' % int(time.time()))


if __name__ == '__main__':

    env = flappy_bird.GameState()

    config = {
        'num_actions': 2,
        'use_gpu': False,
        'epsilon_start': 0.3,
        'epsilon_end': 0.6,
        'epsilon_decay': 0.000001,
        'gamma': 0.99,
        'update_target': 10,
        'epoches': 1000,
        'batch_size': 16,
        'learning_rate': 0.001,
        'replay_memory_size': 2000,
    }

    dueling_dqn = DuelingDQN(config, env)
    dueling_dqn.train()

