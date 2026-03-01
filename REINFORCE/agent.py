import numpy as np
import tensorflow as tf
from model import build_policy_model


class REINFORCE:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        lr: float,
        gamma: float,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        policy_input = tf.keras.Input(
            shape=(self.state_dim,), name="policy_input", dtype=tf.float32
        )
        self.policy = build_policy_model(hidden_dim, action_dim, policy_input)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def take_action(self, state):
        state = tf.reshape(tf.convert_to_tensor(state, dtype=tf.float32), [1, -1])
        probs = self.policy(state).numpy().ravel()
        action = np.random.choice(self.action_dim, p=probs)
        return action

    def _compute_returns(self, rewards):
        returns = []
        discounted_return = 0.0
        for reward in rewards[::-1]:
            discounted_return = reward + self.gamma * discounted_return
            returns.append(discounted_return)
        returns.reverse()
        returns = np.array(returns, dtype=np.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        return returns

    def update(self, states, actions, rewards):
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        returns = self._compute_returns(rewards)

        with tf.GradientTape() as tape:
            probs = self.policy(states, training=True)
            chosen_probs = tf.gather(probs, actions, axis=1, batch_dims=1)
            log_probs = tf.math.log(chosen_probs + 1e-8)
            loss = -tf.reduce_mean(log_probs * returns)

        grads = tape.gradient(loss, self.policy.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy.trainable_variables))
