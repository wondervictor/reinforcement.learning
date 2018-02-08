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

import cv2
import torch
import torch.nn as nn
import numpy as np
from flappy_bird import flappy_bird
import matplotlib.pyplot as plt


class ValueNet(nn.Module):

    def __init__(self):
        super(ValueNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(9 * 16 * 128, 512),
            nn.LeakyReLU(negative_slope=0.03),
            nn.Linear(512, 64),
            nn.LeakyReLU(negative_slope=0.03),
            nn.Linear(64, 32),
            nn.LeakyReLU(negative_slope=0.03),
            nn.Linear(32, 1),
            nn.LeakyReLU(negative_slope=0.03),
        )

    def forward(self, x):
        return self.fc(x)


class AdvantageNet(nn.Module):

    def __init__(self, output_actions):
        super(AdvantageNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(9 * 16 * 128, 512),
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

        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=in_chans, out_channels=32, kernel_size=5, padding=2),
            nn.LeakyReLU(negative_slope=0.03),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.03),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.03),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=128)
        )

        self.advantage_net = AdvantageNet(output_actions)
        self.value_net = ValueNet()

    def forward(self, x):

        x = self.convs(x)
        x = x.view(-1, 9 * 16 * 128)
        value = self.value_net(x)
        advatage = self.advantage_net(x)

        out = value + (advatage - torch.mean(advatage, 1))
        return out


class DuelingDQN(object):

    def __init__(self, env):

        self.env = env

    def apply_actions(self, action):
        pass

    def get_state(self, x):
        img = np.transpose(x, (1, 0, 2))
        img = cv2.resize(img, (144, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def learn(self):

        pass

    def train(self):

        pass


if __name__ == '__main__':

    env = flappy_bird.GameState()

    s = env.frame_step([0, 1])
    screen = s[0]
    screen = np.transpose(screen, (1, 0, 2))
    img = cv2.resize(screen, (144, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('ss.jpg', img)

