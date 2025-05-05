from collections import defaultdict
import gym
import numpy as np


class SARSA:
    def __init__(self, env: gym.Env, epsilon_end: float=0.001, epsilon_start: float=0.95, epsilon_decay: int=200, alpha: float=0.05, gamma: float=0.95):  # 显式传入动作数
        env_name = env.spec.id
        # if env_name != "CliffWalking-v0":
        #     raise ValueError(f"Unsupported environment: {env_name}")
        self.n_actions = env.action_space.n  # 先定义n_actions
        self.Q_table = defaultdict(lambda: np.zeros(self.n_actions))  # 初始化state-action值表,每个state默认的action值为0

        # 超参数
        self.sample_count = 0
        self.epsilon_end = epsilon_end # epsilon终止值
        self.epsilon_start = epsilon_start # epsilon起始值
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.gamma = gamma

    def sample_action(self, state):
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (
            self.epsilon_start - self.epsilon_end
        ) * np.exp(-1.0 * self.sample_count / self.epsilon_decay)

        state_str = str(state)  # 确保状态转换为字符串
        if np.random.uniform(0, 1) > self.epsilon: # epsilon-greedy策略
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
        Q_target = reward + (1 - terminated) * self.gamma * np.max(
            self.Q_table[next_state_str]
        ) # q_learning评估动作函数值时，下一个时间步采取的动作未必是当前评估时下一时间步采用的动作，因此q_learning为off-policy算法
        self.Q_table[state_str][action] += self.alpha * (Q_target - Q_predict)

    def update_sarsa(self, state, action, reward, next_state, next_action, terminated):
        state_str = str(state)
        next_state_str = str(next_state)
        Q_predict = self.Q_table[state_str][action]
        Q_target = (
            reward
            + (1 - terminated) * self.gamma * self.Q_table[next_state_str][next_action]
        ) # sarsa评估动作函数值时，下一个时间步采取的动作一定是当前评估时采用的动作，因此sarsa为on-policy算法，且相对保守
        self.Q_table[state_str][action] += self.alpha * (Q_target - Q_predict)