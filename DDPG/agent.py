from collections import deque
import numpy as np
from model import build_actor_model, build_critic_model
import tensorflow as tf
import gym

class DDPG:
    def __init__(self, env: gym.Env, state_dim: int, action_dim: int, hidden_dim: int, actor_lr: float, critic_lr: float, sigma: float, tau: float, gamma: float, batch_size: int, buffer_size: int):
        self.env = env
        self.env_name = env.spec.name
        
        self.action_bound = env.action_space.high[0]
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor_inputs = tf.keras.layers.Input(shape=(state_dim,), name='actor_input: state')
        self.critic_inputs_state = tf.keras.layers.Input(shape=(state_dim,), name='critic_input: state')
        self.critic_inputs_action = tf.keras.layers.Input(shape=(action_dim,), name='critic_input: action')
        self.actor = build_actor_model(action_dim, hidden_dim, self.action_bound, self.actor_inputs)
        self.actor_target = build_actor_model(action_dim, hidden_dim, self.action_bound, self.actor_inputs)
        self.critic = build_critic_model(hidden_dim, self.critic_inputs_state, self.critic_inputs_action)
        self.critic_target = build_critic_model(hidden_dim, self.critic_inputs_state, self.critic_inputs_action)
        
        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target.set_weights(self.critic.get_weights())
        
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        
        self.sigma = sigma
        self.tau = tau
        self.gamma = gamma
        
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        self.memory = deque(maxlen=self.buffer_size)
    
    def take_action(self, state):
        state = tf.reshape(state, [1, self.state_dim])
        action = self.actor(state).numpy()[0]
        action_final = action + self.sigma * np.random.randn(self.action_dim)
        action_final = np.clip(action_final, self.env.action_space.low, self.env.action_space.high)
        return action_final
    
    def soft_update(self, net: tf.keras.Model, target_net: tf.keras.Model):
        for var, target_var in zip(net.trainable_variables, target_net.trainable_variables):
            target_var.assign(self.tau * var + (1 - self.tau) * target_var)
    
    def update(self, trainsition_dict: dict):
        states = np.array(trainsition_dict['states'], dtype=np.float32)
        actions = np.array(trainsition_dict['actions'], dtype=np.float32)
        rewards = np.array(trainsition_dict['rewards'], dtype=np.float32)
        next_states = np.array(trainsition_dict['next_states'], dtype=np.float32)
        dones = np.array(trainsition_dict['dones'], dtype=np.float32)
        
        next_q_values = self.critic_target([next_states, self.actor_target(next_states)])
        q_targets = rewards + self.gamma * next_q_values * (1 - dones)
        
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            critic_loss = tf.reduce_mean(tf.square(q_targets - self.critic([states, actions])))
            actor_loss = -tf.reduce_mean(self.critic([states, self.actor(states)]))
        critic_gradients = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
        actor_gradients = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        
        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)
