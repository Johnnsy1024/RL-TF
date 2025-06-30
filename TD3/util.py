import random
from collections import deque

import numpy as np
from slots import BATCH_SIZE, BUFFER_SIZE


class ReplayBuffer:
    def __init__(self, size=BUFFER_SIZE):
        self.buffer = deque(maxlen=size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(self, batch_size=BATCH_SIZE):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return (
            state,
            action,
            reward.reshape(-1, 1),
            next_state,
            done.reshape(-1, 1),
        )

    def __len__(self):
        return len(self.buffer)


class OUActionNoise:
    def __init__(self, mean, std_deviation=0.2, theta=0.15, dt=1e-2):
        self.theta = theta
        self.mu = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.reset()

    def __call__(self):
        noise = self.theta * (self.mu - self.x) + self.std_dev * np.random.normal(
            size=self.mu.shape
        )
        self.x += noise * self.dt
        return self.x

    def reset(self):
        self.x = np.zeros_like(self.mu)
