from collections import deque

import gym
import tensorflow as tf
from model import build_actor, build_critic


class SAC:
    def __init__(
        self,
        env: gym.Env,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        actor_lr: float,
        critic_lr: float,
        alpha_lr: float,
        gamma: float,
        tau: float,
        alpha_init: float,
        target_entropy: float,
        buffer_size: int,
    ):
        self.actor = build_actor(
            state_dim, hidden_dim, action_dim, env.action_space.high[0]
        )

        self.critic_1 = build_critic(state_dim, hidden_dim, action_dim)
        self.critic_2 = build_critic(state_dim, hidden_dim, action_dim)

        self.target_critic_1 = build_critic(state_dim, hidden_dim, action_dim)
        self.target_critic_2 = build_critic(state_dim, hidden_dim, action_dim)

        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_1_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.critic_2_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha_init
        self.target_entropy = target_entropy

        self.log_alpha = tf.Variable(tf.math.log(alpha_init), trainable=True)
        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha_lr)

        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.memory = deque(maxlen=buffer_size)

    def take_action(self, state: tf.Tensor):
        state = tf.reshape(state, [-1, self.state_dim])
        action, _ = self.actor(state)
        return action.numpy()[0]

    def calc_target(self, rewards, next_states, dones):
        next_actions, log_probs = self.actor(next_states)
        entropy = -log_probs
        q1_value = self.target_critic_1([next_states, next_actions])
        q2_value = self.target_critic_2([next_states, next_actions])
        next_value = tf.minimum(q1_value, q2_value) + tf.exp(self.log_alpha) * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for var, target_var in zip(net.variables, target_net.variables):
            target_var.assign(self.tau * var + (1 - self.tau) * target_var)

    def update(self, transition_dict: dict):
        states = tf.convert_to_tensor(transition_dict["states"], dtype=tf.float32)
        actions = tf.convert_to_tensor(transition_dict["actions"], dtype=tf.float32)
        rewards = tf.convert_to_tensor(transition_dict["rewards"], dtype=tf.float32)
        next_states = tf.convert_to_tensor(
            transition_dict["next_states"], dtype=tf.float32
        )
        dones = tf.convert_to_tensor(transition_dict["dones"], dtype=tf.float32)

        rewards = (rewards + 8.0) / 8.0  # normalize rewards
        with tf.GradientTape() as critic_1_tape, tf.GradientTape() as critic_2_tape, tf.GradientTape() as actor_tape, tf.GradientTape() as alpha_tape:
            td_target = self.calc_target(rewards, next_states, dones)
            critic_1_loss = tf.reduce_mean(
                tf.square(self.critic_1([states, actions]) - td_target)
            )
            critic_2_loss = tf.reduce_mean(
                tf.square(self.critic_2([states, actions]) - td_target)
            )

            new_actions, log_probs = self.actor(states)
            entropy = -log_probs
            q1_value = self.critic_1([states, new_actions])
            q2_value = self.critic_2([states, new_actions])
            actor_loss = tf.reduce_mean(
                tf.exp(self.log_alpha) * entropy - tf.minimum(q1_value, q2_value)
            )

            alpha_loss = tf.reduce_mean(
                tf.exp(self.log_alpha) * (entropy - self.target_entropy)
            )

        critic_1_grads = critic_1_tape.gradient(
            critic_1_loss, self.critic_1.trainable_variables
        )
        critic_2_grads = critic_2_tape.gradient(
            critic_2_loss, self.critic_2.trainable_variables
        )
        self.critic_1_optimizer.apply_gradients(
            zip(critic_1_grads, self.critic_1.trainable_variables)
        )
        self.critic_2_optimizer.apply_gradients(
            zip(critic_2_grads, self.critic_2.trainable_variables)
        )

        actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grads, self.actor.trainable_variables)
        )

        alpha_grads = alpha_tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)
