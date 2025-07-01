from model import build_actor_model, build_critic_model
from slots import action_dim, state_dim
import tensorflow as tf
import numpy as np


class ActorCritic:
    def __init__(self):
        self.inputs = tf.keras.Input(shape=(state_dim,), name="state")
        self.actor_model = build_actor_model(action_dim, self.inputs)
        self.critic_model = build_critic_model(self.inputs)

    def get_action(self, state):
        state = tf.reshape(state, [-1, state_dim])
        probs = self.actor_model(state)
        action = np.random.choice(action_dim, p=probs.numpy()[0])
        return action
