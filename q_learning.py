from collections import defaultdict
import numpy as np

class Agent:
    def __init__(self, n_actions=4):  # 显式传入动作数
        self.n_actions = n_actions  # 先定义n_actions
        self.Q_table = defaultdict(lambda: np.zeros(self.n_actions))  # 再初始化Q表
        
        # 超参数
        self.sample_count = 0
        self.epsilon_end = 0.001
        self.epsilon_start = 0.95
        self.epsilon_decay = 200
        self.alpha = 0.05
        self.gamma = 0.95

    def sample_action(self, state):
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1. * self.sample_count / self.epsilon_decay)
        
        state_str = str(state)  # 确保状态转换为字符串
        if np.random.uniform(0, 1) > self.epsilon:
            action = np.argmax(self.Q_table[state_str])
        else:
            action = np.random.choice(self.n_actions)
        return action

    def predict_action(self, state):
        state_str = str(state)
        action = np.argmax(self.Q_table[state_str])
        return action
    
    def update(self, state, action, reward, next_state, terminated):
        state_str = str(state)
        next_state_str = str(next_state)
        Q_predict = self.Q_table[state_str][action]
        Q_target = reward + (1 - terminated) * self.gamma * np.max(self.Q_table[next_state_str])
        self.Q_table[state_str][action] += self.alpha * (Q_target - Q_predict)
    
    def update_sarsa(self, state, action, reward, next_state, next_action, terminated):
        state_str = str(state)
        next_state_str = str(next_state)
        Q_predict = self.Q_table[state_str][action]
        Q_target = reward + (1 - terminated) * self.gamma * self.Q_table[next_state_str][next_action]
        self.Q_table[state_str][action] += self.alpha * (Q_target - Q_predict)
import gym
from tqdm import tqdm  # 进度条工具

# 创建环境和智能体
env = gym.make('CliffWalking-v0')
agent = Agent(n_actions=env.action_space.n)

# 训练参数
n_episodes = 1000  # 训练总轮数
print_interval = 100  # 每100轮打印一次结果

# 训练循环
rewards = []  # 记录每轮的总奖励
for episode in range(n_episodes):
    state = env.reset()
    total_reward = 0
    terminated = False
    
    while not terminated:
        action = agent.sample_action(state)
        next_state, reward, terminated, truncated, _  = env.step(action)
        agent.update(state, action, reward, next_state, terminated)
        state = next_state
        total_reward += reward
    
    rewards.append(total_reward)
    
    # 定期打印训练进度
    if episode % print_interval == 0:
        avg_reward = np.mean(rewards[-print_interval:])
        print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")

# 测试训练结果
test_episodes = 10
for episode in range(test_episodes):
    state = env.reset()
    total_reward = 0
    terminated = False
    path = []
    
    while not terminated:
        action = agent.predict_action(state)  # 测试时使用确定性策略
        next_state, reward, terminated, truncated, _  = env.step(action)
        path.append((state, action))
        state = next_state
        total_reward += reward
    
    print(f"Test Episode {episode + 1}:")
    print(f"Total Reward: {total_reward}")
    print(f"Path: {path[:10]}...")  # 只打印前10步避免过长
    print("-----")
import matplotlib.pyplot as plt

# 在训练代码后添加
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Progress')
# plt.show()
plt.savefig('./q_learning.png')