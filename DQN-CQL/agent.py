import numpy as np
import tensorflow as tf
from model import build_dueling_model, build_model
from slots import ACTION_DIM, DQN_TYPE, GAMMA, TARGET_UPDATE_FREQ


class DQN:
    def __init__(self):
        self.dnq_type = DQN_TYPE
        self.q_network = build_model(ACTION_DIM)
        self.target_network = build_model(ACTION_DIM)
        self.target_network.set_weights(self.q_network.get_weights())

    def get_action(self, state, epsilon):
        state = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        if np.random.rand() < epsilon:
            return np.random.randint(ACTION_DIM)
        q_values = self.q_network.predict(state, verbose=0)
        return np.argmax(q_values[0])

    @tf.function
    def update(self, states, actions, rewards, next_states, dones):
        with tf.GradientTape() as tape:
            q_values = self.q_network(states, training=True)
            q_values_w_action = tf.gather(q_values, actions, axis=1, batch_dims=1)
            if self.dnq_type == "dqn" or self.dnq_type == "dueling_dqn":
                max_next_q_values = tf.reduce_max(
                    self.target_network(next_states, training=False),
                    1,
                    keepdims=True,
                )
            elif self.dnq_type == "double_dqn":
                max_next_action = tf.argmax(
                    self.q_network(next_states, training=False), axis=1
                )
                next_q_values_target = self.target_network(next_states, training=False)
                max_next_q_values = tf.gather(
                    next_q_values_target, max_next_action[:, None], axis=1, batch_dims=1
                )
            q_target = rewards + GAMMA * max_next_q_values * (1 - dones)
            logsumexpq = tf.reduce_logsumexp(q_values, axis=1)
            q_dataset_actions = tf.reduce_sum(
                tf.reduce_sum(q_values * tf.one_hot(actions, ACTION_DIM), axis=-1),
                axis=-1,
            )
            conservative_loss = tf.reduce_mean(logsumexpq - q_dataset_actions)
            q_loss = (
                tf.reduce_mean(tf.square(q_target - q_values_w_action))
                + conservative_loss * 0.02
            )
        grads = tape.gradient(q_loss, self.q_network.trainable_variables)
        self.q_network.optimizer.apply_gradients(
            zip(grads, self.q_network.trainable_variables)
        )

    def update_target_network(self, sample_cnt: int):
        if sample_cnt % TARGET_UPDATE_FREQ == 0:
            self.target_network.set_weights(self.q_network.get_weights())


class DuelingDQN(DQN):
    def __init__(self):
        super().__init__()
        self.q_network = build_dueling_model(ACTION_DIM)
        self.target_network = build_dueling_model(ACTION_DIM)
        self.target_network.set_weights(self.q_network.get_weights())
