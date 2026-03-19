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
            action.reshape(-1, 1),
            reward.reshape(-1, 1),  # 保证reward数值维度为(batch_size, 1)
            next_state,
            done.reshape(-1, 1),  # 保证done数值维度为(batch_size, 1)
        )

    def __len__(self):
        return len(self.buffer)
