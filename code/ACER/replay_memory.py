#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/5/25 下午8:31
# @Author  : Yang Yuchi

import random
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'policy', 'is_done'))


class ReplayMemory(object):
    def __init__(self, max_episodes, max_epi_length):
        self.buffer = deque(maxlen=max_episodes)
        self.buffer.append(deque(maxlen=max_epi_length))
        self.max_epi_length = max_epi_length
        self.index = 0

    def __len__(self):
        return sum(len(episode) for episode in self.buffer)

    def push(self, state, action, reward, policy, is_done):
        self.buffer[self.index].append(Transition(state, action, reward, policy, is_done))
        if is_done:
            if len(self.buffer[self.index]) < 5:
                self.buffer.pop()
                self.buffer.append(deque(maxlen=self.max_epi_length))
            else:
                self.buffer.append(deque(maxlen=self.max_epi_length))
                self.index = min(self.index + 1, self.buffer.maxlen - 1)

    def sample(self, batch_size):
        # check if the last item in buffer is empty
        if self.buffer[len(self.buffer) - 1]:
            length = len(self.buffer)
        else:
            length = len(self.buffer) - 1
        episode = self.buffer[random.randrange(length)]  # pick an episode randomly
        if len(episode) <= batch_size:
            return list(episode)
        else:
            i = random.randrange(len(episode) - batch_size + 1)
            samples = list(episode)[i:i + batch_size]
            return samples

