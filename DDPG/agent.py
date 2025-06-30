from collections import deque

import gym
import numpy as np
import tensorflow as tf
from model import build_actor_model, build_critic_model
from slots import NOISE_TYPE


class DDPG:
    def __init__(
        self,
        env: gym.Env,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        actor_lr: float,
        critic_lr: float,
        sigma: float,
        sigma_end: float,
        tau: float,
        gamma: float,
        noise_type: str = "normal",
    ):
        if noise_type not in ["normal", "ou"]:
            raise ValueError("noise_type must be 'normal' or 'ou'")
        self.noise_type = noise_type
        self.env = env
        self.env_name = env.spec.name

        self.action_bound = env.action_space.high[0]
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = build_actor_model(
            state_dim, action_dim, hidden_dim, self.action_bound
        )
        self.actor_target = build_actor_model(
            state_dim, action_dim, hidden_dim, self.action_bound
        )
        self.critic = build_critic_model(state_dim, action_dim, hidden_dim)
        self.critic_target = build_critic_model(state_dim, action_dim, hidden_dim)

        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

        self.sigma = sigma
        self.sigma_end = sigma_end
        self.tau = tau
        self.gamma = gamma

    def take_action(self, state):
        state = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        action = self.actor(state)[0].numpy()
        return action

    @tf.function
    def update(self, state, action, reward, next_state, done):
        # 更新critic
        with tf.GradientTape() as tape:
            target_actions = self.actor_target(next_state)
            target_q = self.critic_target([next_state, target_actions])
            target_q = reward + self.gamma * (1 - done) * target_q
            critic_value = self.critic([state, action])
            critic_loss = tf.reduce_mean(tf.square(target_q - critic_value))
        grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
        # 更新actor
        with tf.GradientTape() as tape:
            actions = self.actor(state)
            critic_value = self.critic([state, actions])
            actor_loss = -tf.reduce_mean(critic_value)
        grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
        # 软更新目标网络
        for var, target_var in zip(self.actor.variables, self.actor_target.variables):
            target_var.assign(self.tau * var + (1 - self.tau) * target_var)
        for var, target_var in zip(self.critic.variables, self.critic_target.variables):
            target_var.assign(self.tau * var + (1 - self.tau) * target_var)
