import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import rl_utils
from loguru import logger
import tensorflow as tf




class PolicyNet(tf.keras.Model):
    """输出不同动作的概率

    Args:
        torch (_type_): _description_
    """
    def __init__(self, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.fc2 = tf.keras.layers.Dense(action_dim, activation='softmax')

    def call(self, x):
        x = self.fc1(x)
        return self.fc2(x)


class ValueNet(tf.keras.Model):
    """输出状态价值

    Args:
        torch (_type_): _description_
    """
    def __init__(self, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.fc2 = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.fc1(x)
        return self.fc2(x)


class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device):
        self.actor = PolicyNet(hidden_dim, action_dim).to(device)
        self.critic = ValueNet(hidden_dim).to(device)
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).view(1, -1)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs) # 构建离散分布
        action = action_dist.sample()
        return action.item()

    def update(self, transition_dict):
        states = tf.convert_to_tensor(transition_dict['states'], dtype=tf.float32)
        actions = tf.convert_to_tensor(transition_dict['actions'], dtype=tf.int32)
        
        rewards = tf.convert_to_tensor(transition_dict['rewards'], dtype=tf.float32)
        next_states = tf.convert_to_tensor(transition_dict['next_states'], dtype=tf.float32)
        
        dones = tf.convert_to_tensor(transition_dict['dones'], dtype=tf.float32)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda, td_delta.numpy())
        old_log_probs = tf.math.log(self.actor(states).gather(1, actions)).numpy()

        for _ in range(self.epochs):
            with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
                logits = self.actor(states)
                probs = tf.gather(logits, actions, axis=1, batch_dims=1)
                log_prbos = tf.math.log(probs + 1e-8)
                ratio = tf.math.exp(log_prbos - old_log_probs)
                surr1 = ratio * advantage
                surr2 = tf.clip_by_value(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
                actor_loss = -tf.math.reduce_mean(tf.math.minimum(surr1, surr2))  # PPO损失函数
                critic_loss = tf.math.reduce_mean(tf.math.square(self.critic(states) - td_target))
            actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
            # log_probs = tf.math.log(self.actor(states).gather(1, actions))
            # ratio = tf.math.exp(log_probs - old_log_probs)
            # surr1 = ratio * advantage
            # surr2 = tf.clip_by_value(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
            # actor_loss = tf.math.reduce_mean(-tf.math.minimum(surr1, surr2))  # PPO损失函数
            # critic_loss = tf.math.reduce_mean(tf.math.square(self.critic(states) - td_target))
            # self.actor_optimizer.minimize(actor_loss)
            # self.critic_optimizer.minimize(critic_loss)
            # self.actor_optimizer.step()
            # self.critic_optimizer.step()

# 超参数
actor_lr = 1e-3
critic_lr = 1e-2
num_episodes = 500
hidden_dim = 128
gamma = 0.98
lmbda = 0.95
epochs = 10
eps = 0.2
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

env_name = 'CliffWalking-v0'
env = gym.make(env_name)
# env.seed(0)
torch.manual_seed(0)
state_dim = env.observation_space.n
action_dim = env.action_space.n
agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
            epochs, eps, gamma, device)

return_list = rl_utils.train_on_policy_agent(env, agent, num_episodes)
# actor_lr = 1e-3
# critic_lr = 1e-2
# num_episodes = 500
# hidden_dim = 128
# gamma = 0.98
# lmbda = 0.95
# epochs = 10
# eps = 0.2
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
#     "cpu")

# env_name = 'CartPole-v0'
# env = gym.make(env_name)
# # env.seed(0)
# torch.manual_seed(0)
# state_dim = env.observation_space.shape[0]
# action_dim = env.action_space.n
# agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
#             epochs, eps, gamma, device)


# return_list = []

# for i in range(num_episodes):
#     state = env.reset()[0]
#     done = False
#     episode_return = 0
#     transition_dict = {
#         'states': [],
#         'actions': [],
#         'rewards': [],
#         'next_states': [],
#         'dones': []
#     }

#     while not done:
#         action = agent.take_action(state)
#         next_state, reward, done, trun, _ = env.step(action)
#         transition_dict['states'].append(state)
#         transition_dict['actions'].append(action)
#         transition_dict['rewards'].append(reward)
#         transition_dict['next_states'].append(next_state)
#         transition_dict['dones'].append(done)

#         state = next_state
#         episode_return += reward

#     return_list.append(episode_return)
#     agent.update(transition_dict)

#     if (i + 1) % 10 == 0:
#         avg_return = np.mean(return_list[-10:])
#         print(f'Episode {i + 1}, Average Return: {avg_return:.2f}')

# # 绘图
# plt.plot(range(len(return_list)), return_list)
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('PPO on ' + env_name)
# plt.show()

episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))
plt.show()

mv_return = rl_utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('PPO on {}'.format(env_name))
plt.show()