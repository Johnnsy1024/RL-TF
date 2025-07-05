from collections import deque

import gym
import numpy as np
import tensorflow as tf
from model import build_actor, build_critic
from slots import env_action_type


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
        self.actor = build_actor(state_dim, hidden_dim, action_dim, env_action_type)

        self.critic_1 = build_critic(state_dim, hidden_dim, action_dim)
        self.critic_2 = build_critic(state_dim, hidden_dim, action_dim)

        self.target_critic_1 = build_critic(state_dim, hidden_dim, action_dim)
        self.target_critic_2 = build_critic(state_dim, hidden_dim, action_dim)

        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_1_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.critic_2_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)

        self.log_alpha = tf.Variable(tf.math.log(alpha_init), trainable=True)
        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=alpha_lr)
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha_init
        self.target_entropy = target_entropy

        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.action_bound = (
            env.action_space.high[0] if env_action_type == "continuous" else None
        )
        self.memory = deque(maxlen=buffer_size)

    def take_action(self, state: tf.Tensor, deterministic: bool = False):
        state = tf.convert_to_tensor(
            tf.reshape(state, [-1, self.state_dim]), dtype=tf.float32
        )
        mean, log_std = self.actor(state)
        log_std = tf.clip_by_value(log_std, -20, 2)
        std = tf.exp(log_std)
        if deterministic:
            action = tf.tanh(mean)
        else:
            normal_sample = tf.random.normal(shape=mean.shape)
            action = tf.tanh(mean + std * normal_sample)
        return (action * self.action_bound).numpy()[0]

    def take_action_discrete(self, state: tf.Tensor, deterministic: bool = False):
        state = tf.convert_to_tensor(
            tf.reshape(state, [-1, self.state_dim]), dtype=tf.float32
        )
        logits = self.actor(state)
        if deterministic:
            action = tf.argmax(logits, axis=-1)
        else:
            action = tf.random.categorical(logits, 1)[0, 0]
        return action.numpy()

    def update(self, transition_dict: dict):
        states = tf.convert_to_tensor(transition_dict["states"], dtype=tf.float32)
        actions = tf.convert_to_tensor(transition_dict["actions"], dtype=tf.float32)
        rewards = tf.convert_to_tensor(transition_dict["rewards"], dtype=tf.float32)
        next_states = tf.convert_to_tensor(
            transition_dict["next_states"], dtype=tf.float32
        )
        dones = tf.convert_to_tensor(transition_dict["dones"], dtype=tf.float32)

        next_mean, next_log_std = self.actor(next_states)
        next_log_std = tf.clip_by_value(next_log_std, -20, 2)
        next_std = tf.exp(next_log_std)
        next_noise = tf.random.normal(shape=next_mean.shape)
        next_action = tf.tanh(next_mean + next_std * next_noise)

        next_log_prob = -0.5 * (
            (next_noise**2 + 2 * next_log_std + tf.math.log(2 * np.pi))
        )
        next_log_prob = tf.reduce_sum(next_log_prob, axis=-1, keepdims=True)
        next_log_prob -= tf.reduce_sum(
            tf.math.log(1 - tf.square(next_action) + 1e-6), axis=-1, keepdims=True
        )
        target_q1 = self.target_critic_1([next_states, next_action])
        target_q2 = self.target_critic_2([next_states, next_action])
        target_q = (
            tf.minimum(target_q1, target_q2) - tf.exp(self.log_alpha) * next_log_prob
        )
        y = rewards + self.gamma * (1 - dones) * target_q

        with tf.GradientTape() as tape1:
            q1_pred = self.critic_1([states, actions])
            critic_loss1 = tf.reduce_mean(tf.square(y - q1_pred))
        grads1 = tape1.gradient(critic_loss1, self.critic_1.trainable_variables)
        self.critic_1_optimizer.apply_gradients(
            zip(grads1, self.critic_1.trainable_variables)
        )

        with tf.GradientTape() as tape2:
            q2_pred = self.critic_2([states, actions])
            critic_loss2 = tf.reduce_mean(tf.square(y - q2_pred))
        grads2 = tape2.gradient(critic_loss2, self.critic_2.trainable_variables)
        self.critic_2_optimizer.apply_gradients(
            zip(grads2, self.critic_2.trainable_variables)
        )

        with tf.GradientTape() as tape3:
            mean, log_std = self.actor(states)
            log_std = tf.clip_by_value(log_std, -20, 2)
            std = tf.exp(log_std)
            noise = tf.random.normal(shape=mean.shape)
            new_action = tf.tanh(mean + std * noise)

            log_prob = -0.5 * ((noise**2 + 2 * log_std + tf.math.log(2 * np.pi)))
            log_prob = tf.reduce_sum(log_prob, axis=-1, keepdims=True)
            log_prob -= tf.reduce_sum(
                tf.math.log(1 - tf.square(new_action) + 1e-6), axis=-1, keepdims=True
            )
            q1_new = self.critic_1([states, new_action])
            policy_loss = tf.reduce_mean(tf.exp(self.log_alpha) * log_prob - q1_new)
        grads3 = tape3.gradient(policy_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads3, self.actor.trainable_variables))

        with tf.GradientTape() as tape4:
            alpha_loss = tf.reduce_mean(
                -self.log_alpha * (tf.stop_gradient(log_prob + self.target_entropy))
            )
        grads4 = tape4.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(grads4, [self.log_alpha]))

        for target_var, var in zip(
            self.target_critic_1.variables, self.critic_1.variables
        ):
            target_var.assign(self.tau * var + (1 - self.tau) * target_var)
        for target_var, var in zip(
            self.target_critic_2.variables, self.critic_2.variables
        ):
            target_var.assign(self.tau * var + (1 - self.tau) * target_var)

    def update_discrete(self, transition_dict: dict):
        states = tf.convert_to_tensor(transition_dict["states"], dtype=tf.float32)
        actions = tf.convert_to_tensor(
            transition_dict["actions"], dtype=tf.int32
        )  # 离散动作
        rewards = tf.convert_to_tensor(transition_dict["rewards"], dtype=tf.float32)
        next_states = tf.convert_to_tensor(
            transition_dict["next_states"], dtype=tf.float32
        )
        dones = tf.convert_to_tensor(transition_dict["dones"], dtype=tf.float32)

        # ========== 1. 下一个状态的动作策略分布 ==========
        logits = self.actor(next_states)  # shape: (batch, action_dim)
        probs = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        # 计算 V(s') = sum_a π(a|s') * (Q(s', a) - α * log π(a|s'))
        batch_size = tf.shape(states)[0]

        # 重复 next_states -> 每个动作都评估一次 Q(s', a)
        next_states_tiled = tf.repeat(next_states, repeats=self.action_dim, axis=0)
        all_actions_onehot = tf.tile(tf.eye(self.action_dim), [batch_size, 1])

        target_q1 = self.target_critic_1([next_states_tiled, all_actions_onehot])
        target_q2 = self.target_critic_2([next_states_tiled, all_actions_onehot])
        target_q = tf.minimum(target_q1, target_q2)
        target_q = tf.reshape(target_q, [batch_size, self.action_dim])

        entropy_term = tf.exp(self.log_alpha) * log_probs
        v_next = tf.reduce_sum(probs * (target_q - entropy_term), axis=1, keepdims=True)

        y = rewards + self.gamma * (1 - dones) * v_next

        # ========== 2. Critic 更新 ==========
        actions_onehot = tf.one_hot(actions, depth=self.action_dim, dtype=tf.float32)

        with tf.GradientTape() as tape1:
            q1 = self.critic_1([states, actions_onehot])
            critic_loss1 = tf.reduce_mean(tf.square(y - q1))
        grads1 = tape1.gradient(critic_loss1, self.critic_1.trainable_variables)
        self.critic_1_optimizer.apply_gradients(
            zip(grads1, self.critic_1.trainable_variables)
        )

        with tf.GradientTape() as tape2:
            q2 = self.critic_2([states, actions_onehot])
            critic_loss2 = tf.reduce_mean(tf.square(y - q2))
        grads2 = tape2.gradient(critic_loss2, self.critic_2.trainable_variables)
        self.critic_2_optimizer.apply_gradients(
            zip(grads2, self.critic_2.trainable_variables)
        )

        # ========== 3. Actor 策略更新 ==========
        with tf.GradientTape() as tape3:
            logits = self.actor(states)
            probs = tf.nn.softmax(logits, axis=-1)
            log_probs = tf.nn.log_softmax(logits, axis=-1)

            # 评估 Q(s, a) for all a
            states_tiled = tf.repeat(states, repeats=self.action_dim, axis=0)
            all_actions_onehot = tf.tile(tf.eye(self.action_dim), [batch_size, 1])
            q1_values = self.critic_1([states_tiled, all_actions_onehot])
            q1_values = tf.reshape(q1_values, [batch_size, self.action_dim])

            # Policy loss: maximize E_a[Q - alpha * log π]
            policy_loss = tf.reduce_mean(
                tf.reduce_sum(
                    probs * (tf.exp(self.log_alpha) * log_probs - q1_values), axis=1
                )
            )
        grads3 = tape3.gradient(policy_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(grads3, self.actor.trainable_variables))

        # ========== 4. Alpha (entropy coefficient) 更新 ==========
        with tf.GradientTape() as tape4:
            entropy = -tf.reduce_sum(probs * log_probs, axis=1, keepdims=True)
            alpha_loss = -tf.reduce_mean(self.log_alpha * (entropy + self.target_entropy))
        grads4 = tape4.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(grads4, [self.log_alpha]))

        # ========== 5. Target 网络软更新 ==========
        for target_var, var in zip(
            self.target_critic_1.variables, self.critic_1.variables
        ):
            target_var.assign(self.tau * var + (1 - self.tau) * target_var)
        for target_var, var in zip(
            self.target_critic_2.variables, self.critic_2.variables
        ):
            target_var.assign(self.tau * var + (1 - self.tau) * target_var)
