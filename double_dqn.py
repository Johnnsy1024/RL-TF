import random
from collections import deque

import gym
import numpy as np
import tensorflow as tf

# 环境初始化
env = gym.make("CartPole-v1")
num_actions = env.action_space.n
state_shape = env.observation_space.shape

# 超参数
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 64
train_start = 1000
memory_size = 10000
target_update_freq = 100

# 经验回放缓存
memory = deque(maxlen=memory_size)


# 构建Q网络
def build_q_network():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(24, activation="relu", input_shape=state_shape),
            tf.keras.layers.Dense(24, activation="relu"),
            tf.keras.layers.Dense(num_actions),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse"
    )
    return model


# 初始化主网络和目标网络
q_network = build_q_network()
target_network = build_q_network()
target_network.set_weights(q_network.get_weights())


# ε-贪婪策略
def get_action(state, epsilon):
    if np.random.rand() <= epsilon:
        return random.randrange(num_actions)
    q_values = q_network.predict(state[np.newaxis], verbose=0)
    return np.argmax(q_values[0])


# Double DQN训练函数
def train():
    if len(memory) < train_start:
        return

    minibatch = random.sample(memory, batch_size)
    states = np.array([transition[0] for transition in minibatch])
    actions = np.array([transition[1] for transition in minibatch])
    rewards = np.array([transition[2] for transition in minibatch])
    next_states = np.array([transition[3] for transition in minibatch])
    dones = np.array([transition[4] for transition in minibatch])

    target_q = q_network.predict(states, verbose=0)
    next_q_main = q_network.predict(next_states, verbose=0)
    next_q_target = target_network.predict(next_states, verbose=0)

    for i in range(batch_size):
        if dones[i]:
            target_q[i][actions[i]] = rewards[i]
        else:
            best_action = np.argmax(next_q_main[i])  # 主网络选动作
            target_q_value = next_q_target[i][best_action]  # 目标网络估值
            target_q[i][actions[i]] = rewards[i] + gamma * target_q_value

    q_network.fit(states, target_q, epochs=1, verbose=0)


# 主循环
num_episodes = 500
steps = 0

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    for t in range(500):
        action = get_action(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        memory.append((state, action, reward, next_state, done))

        state = next_state
        total_reward += reward
        steps += 1

        train()

        if steps % target_update_freq == 0:
            target_network.set_weights(q_network.get_weights())

        if done:
            print(
                f"Episode {episode+1}: Reward = {total_reward}, Epsilon = {epsilon:.3f}"
            )
            break

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
