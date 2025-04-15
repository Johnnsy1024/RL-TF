import numpy as np


# 定义网格世界环境
class GridWorld:
    def __init__(self):
        self.size = 4
        self.num_states = self.size * self.size
        self.num_actions = 4  # 0:上, 1:下, 2:左, 3:右
        self.goal = (3, 3)  # 目标位置
        self.blocks = [(1, 1)]  # 障碍物位置

        # 定义转移概率（确定性环境）
        self.transition_prob = 1.0

        # 定义奖励函数
        self.rewards = np.zeros((self.size, self.size))
        self.rewards[self.goal] = 1.0  # 到达目标奖励+1

    def get_states(self):
        return [(i, j) for i in range(self.size) for j in range(self.size)]

    def get_actions(self, state):
        if state == self.goal:  # 终止状态没有可用动作
            return []
        return list(range(self.num_actions))

    def step(self, state, action):
        if state == self.goal:
            return state, 0

        next_state = list(state)
        if action == 0:  # 上
            next_state[0] = max(0, state[0] - 1)
        elif action == 1:  # 下
            next_state[0] = min(self.size - 1, state[0] + 1)
        elif action == 2:  # 左
            next_state[1] = max(0, state[1] - 1)
        elif action == 3:  # 右
            next_state[1] = min(self.size - 1, state[1] + 1)

        next_state = tuple(next_state)
        if next_state in self.blocks:  # 遇到障碍物保持原地
            next_state = state

        reward = self.rewards[next_state]
        return next_state, reward


# 值迭代算法
def value_iteration(env, gamma=0.9, theta=1e-6):
    values = np.zeros((env.size, env.size))
    policy = np.zeros((env.size, env.size), dtype=int)

    while True:
        delta = 0
        for state in env.get_states():
            if state == env.goal:
                continue
            v = values[state]
            max_value = -float("inf")  # 对每一个state都预设一个state-value
            for action in env.get_actions(state):  # 遍历action,找到最大的state-value
                next_state, reward = env.step(state, action)
                value = reward + gamma * values[next_state]
                if value > max_value:
                    max_value = value
            values[state] = max_value
            delta = max(delta, abs(v - values[state]))

        if delta < theta:
            break
    # 直到全部state的value都收敛了之后
    # 提取最优策略
    for state in env.get_states():
        if state == env.goal:
            continue
        max_value = -float("inf")
        best_action = 0
        for action in env.get_actions(state):
            next_state, reward = env.step(state, action)
            value = reward + gamma * values[next_state]  # 此时的values已经收敛到最优
            if value > max_value:
                max_value = value
                best_action = action
        policy[state] = best_action

    return values, policy


# 策略迭代算法
def policy_iteration(env, gamma=0.9, theta=1e-6):
    # 初始化随机策略
    policy = np.random.choice(env.num_actions, size=(env.size, env.size))
    values = np.zeros((env.size, env.size))

    while True:
        # 策略评估
        while True:
            delta = 0
            for state in env.get_states():
                if state == env.goal:
                    continue

                v = values[state]
                action = policy[state]
                next_state, reward = env.step(state, action)
                values[state] = reward + gamma * values[next_state]
                delta = max(delta, abs(v - values[state]))

            if delta < theta:
                break

        # 策略改进
        policy_stable = True
        for state in env.get_states():
            if state == env.goal:
                continue

            old_action = policy[state]
            max_value = -float("inf")
            best_action = old_action

            for action in env.get_actions(state):
                next_state, reward = env.step(state, action)
                value = reward + gamma * values[next_state]
                if value > max_value:
                    max_value = value
                    best_action = action

            policy[state] = best_action
            if best_action != old_action:
                policy_stable = False

        if policy_stable:
            break

    return values, policy


# 运行算法并显示结果
env = GridWorld()

print("Value Iteration Results:")
vi_values, vi_policy = value_iteration(env)
print("Value Function:")
print(vi_values)
print("Policy (0:上, 1:下, 2:左, 3:右):")
print(vi_policy)

print("\nPolicy Iteration Results:")
pi_values, pi_policy = policy_iteration(env)
print("Value Function:")
print(pi_values)
print("Policy (0:上, 1:下, 2:左, 3:右):")
print(pi_policy)
