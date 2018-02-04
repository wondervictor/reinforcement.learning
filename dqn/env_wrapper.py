# -*- coding: utf-8 -*-

"""
Environment Wrapper
"""

import gym
import matplotlib.pyplot as plt


class MountainCar(object):

    def __init__(self):
        self.env = gym.make('MountainCar-v0').unwrapped


    def get_screen(self):
        pass


    def apply_actions(self, actions):

        self.env.step(actions)

