import numpy as np
import tensorflow as tf
import random
from collections import deque
import gymnasium as gym
import matplotlib.pyplot as plt
# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# 创建 CliffWalking 环境
env = gym.make("CliffWalking-v0")

# 超参数
learning_rate = 0.001
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995
batch_size = 64
max_episodes = 500
target_update_freq = 10
memory_size = 10000

# 获取状态和动作空间
state_size = env.observation_space.n
action_size = env.action_space.n

# 将离散状态转换为 one-hot 编码
def one_hot(state, state_size):
    return np.identity(state_size)[state:state+1]

# 构建 Q 网络
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(state_size,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(action_size)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss='mse')
    return model

# 初始化 Q 网络和目标网络
q_network = build_model()
target_network = build_model()
target_network.set_weights(q_network.get_weights())

# 经验回放缓冲区
memory = deque(maxlen=memory_size)

# 选择动作（epsilon-greedy）
def get_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(action_size)
    q_values = q_network.predict(state, verbose=0)
    return np.argmax(q_values[0])
reward_list = []
# 训练 DQN
for episode in range(max_episodes):
    state, _ = env.reset()
    state = one_hot(state, state_size)
    total_reward = 0
    done = False

    while not done:
        action = get_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state_one_hot = one_hot(next_state, state_size)

        memory.append((state, action, reward, next_state_one_hot, done))
        state = next_state_one_hot
        total_reward += reward

        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = np.vstack(states)
            next_states = np.vstack(next_states)
            q_values = q_network.predict(states, verbose=0)
            next_q_values = target_network.predict(next_states, verbose=0)

            for i in range(batch_size):
                target = rewards[i]
                if not dones[i]:
                    target += gamma * np.max(next_q_values[i])
                q_values[i][actions[i]] = target

            q_network.fit(states, q_values, epochs=1, verbose=0)

    # 更新目标网络
    if episode % target_update_freq == 0:
        target_network.set_weights(q_network.get_weights())

    # 衰减 epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    reward_list.append(total_reward)
    print(f"Episode {episode+1}/{max_episodes} - Total Reward: {total_reward} - Epsilon: {epsilon:.3f}")

env.close()
plt.plot(reward_list)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()
