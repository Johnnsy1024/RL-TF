import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gym

# 创建环境
env = gym.make('CartPole-v1')

# 超参数
gamma = 0.99
learning_rate = 0.001
episodes = 500

# 策略网络 (Actor)
class ActorNetwork(tf.keras.Model):
    def __init__(self):
        super(ActorNetwork, self).__init__()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(2, activation='softmax')  # 2个动作

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

# 值网络 (Critic)
class CriticNetwork(tf.keras.Model):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(1)  # 输出一个值

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

# 创建策略和值网络
actor_net = ActorNetwork()
critic_net = CriticNetwork()

actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate)

def select_action(state):
    state = tf.convert_to_tensor(state, dtype=tf.float32)
    probs = actor_net(state[None, :])
    action = np.random.choice(2, p=probs.numpy()[0])
    return action, probs[0, action]

def compute_return(rewards, gamma):
    returns = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = rewards[t] + gamma * running_add
        returns[t] = running_add
    return returns

def train_step(states, actions, returns, advantages):
    with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
        actor_loss = 0
        critic_loss = 0
        for i in range(len(states)):
            state = tf.convert_to_tensor(states[i], dtype=tf.float32)
            action = actions[i]
            return_ = returns[i]
            advantage = advantages[i]
            
            # 计算Actor损失
            probs = actor_net(state[None, :])
            action_prob = probs[0, action]
            actor_loss -= tf.math.log(action_prob) * advantage
            
            # 计算Critic损失
            value = critic_net(state[None, :])
            critic_loss += tf.square(return_ - value)
        
        # 更新网络
        actor_gradients = actor_tape.gradient(actor_loss, actor_net.trainable_variables)
        critic_gradients = critic_tape.gradient(critic_loss, critic_net.trainable_variables)
        
        actor_optimizer.apply_gradients(zip(actor_gradients, actor_net.trainable_variables))
        critic_optimizer.apply_gradients(zip(critic_gradients, critic_net.trainable_variables))

# 训练过程
for episode in range(episodes):
    state = env.reset()[0]
    episode_rewards = []
    states, actions, rewards, values = [], [], [], []

    done = False
    while not done:
        action, probs = select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        value = critic_net(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
        values.append(value.numpy())
        
        state = next_state
        episode_rewards.append(reward)

    returns = compute_return(rewards, gamma)
    advantages = np.array(returns) - np.array(values).flatten()
    train_step(states, actions, returns, advantages)

    if episode % 50 == 0:
        print(f'Episode {episode}, Total Reward: {np.sum(episode_rewards)}')

env.close()
