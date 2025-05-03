import numpy as np
import tensorflow as tf
import gym
from model import build_actor_model, build_critic_model
from loguru import logger
class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, env: gym.Env, hidden_dim: int, state_dim: int, action_dim: int, actor_lr: float, critic_lr: float, lmbda: float, epochs: int, eps: float, gamma: float):
        self.state_dim = state_dim
        actor_input = tf.keras.Input(shape=(self.state_dim,), name='actor_input', dtype=tf.float32)
        critic_input = tf.keras.Input(shape=(self.state_dim,), name='critic_input', dtype=tf.float32)
        self.actor = build_actor_model(hidden_dim, action_dim, actor_input)
        self.critic = build_critic_model(hidden_dim, critic_input)

        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数

    def take_action(self, state):
        state = tf.reshape(state, [-1, self.state_dim])
        probs = self.actor(state)
        action = np.random.choice(probs.shape[1], p=probs.numpy().ravel())
        return action

    def compute_advantage(self, gamma: float, lmbda: float, td_delta):
        advantage = 0.0
        advantage_list = []
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return np.array(advantage_list, dtype=np.float32)
    
    def update(self, transition_dict):
        states = np.array(transition_dict['states'], dtype=np.float32)
        actions = np.array(transition_dict['actions'])
        rewards = np.array(transition_dict['rewards'], dtype=np.float32)
        next_states = np.array(transition_dict['next_states'], dtype=np.float32)
        dones = np.array(transition_dict['dones'], dtype=np.float32)
        values = self.critic(states).numpy().squeeze()
        next_values = self.critic(next_states).numpy().squeeze()
        td_target = rewards + self.gamma * next_values * (1 - dones)
        td_delta = td_target - values
        advantage = self.compute_advantage(self.gamma, self.lmbda, td_delta)
        # probs = self.actor(states).numpy()
        # old_probs = probs[np.arange(len(actions)), actions]
        # old_log_probs = np.log(old_probs + 1e-8)
        old_log_probs = tf.math.log(tf.gather(self.actor(states), actions, axis=1, batch_dims=1) + 1e-8).numpy()

        for _ in range(self.epochs):
            with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
                logits = self.actor(states)
                probs = tf.gather(logits, actions, axis=1, batch_dims=1)
                log_probs = tf.math.log(probs + 1e-8)
                ratio = tf.exp(log_probs - old_log_probs)
                surr1 = ratio * advantage
                surr2 = tf.clip_by_value(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
                actor_loss = -tf.math.reduce_mean(tf.minimum(surr1, surr2))  # PPO损失函数
                value_preds = tf.squeeze(self.critic(states), axis=1)
                critic_loss = tf.math.reduce_mean(tf.square(td_target - value_preds))
            actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
