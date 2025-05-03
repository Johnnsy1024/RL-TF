from model import build_model, build_dueling_model
from slots import action_dim, state_dim, memory_size, env, target_update_freq
from collections import deque
import tensorflow as tf
import numpy as np

class DQN:
    def __init__(self):
        self.memory = deque(maxlen=memory_size)
        self.inputs = tf.keras.Input(shape=(state_dim,))
        self.q_network = build_model(action_dim, self.inputs)
        self.target_network = build_model(action_dim, self.inputs)
        self.target_network.set_weights(self.q_network.get_weights())
    
    def get_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(action_dim)
        q_values = self.q_network.predict(state, verbose=0)
        return np.argmax(q_values[0])
    
    def update_target_network(self, episode: int):
        if episode % target_update_freq == 0:
            self.target_network.set_weights(self.q_network.get_weights())

class DuelingDQN(DQN):
    def __init__(self):
        super().__init__()
        self.q_network = build_dueling_model(action_dim, self.inputs)
        self.target_network = build_dueling_model(action_dim, self.inputs)
        self.target_network.set_weights(self.q_network.get_weights())