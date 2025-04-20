import numpy as np
import tensorflow as tf
import gym

# 创建环境
env = gym.make('CartPole-v1')

# 超参数
gamma = 0.99
learning_rate = 0.01
episodes = 1000

# 策略网络
class PolicyNetwork(tf.keras.Model):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(2, activation='softmax')  # 2个动作

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

# 创建策略网络
policy_net = PolicyNetwork()
optimizer = tf.keras.optimizers.Adam(learning_rate)

def select_action(state):
    state = tf.convert_to_tensor(state, dtype=tf.float32)
    probs = policy_net(state[None, :])
    action = np.random.choice(2, p=probs.numpy()[0])
    return action, probs[0, action]

def compute_return(rewards, gamma):
    returns = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = rewards[t] + gamma * running_add
        returns[t] = running_add
    return returns

def train_step(states, actions, returns):
    with tf.GradientTape() as tape:
        loss = 0
        for i in range(len(states)):
            state = tf.convert_to_tensor(states[i], dtype=tf.float32)
            action = actions[i]
            return_ = returns[i]
            
            probs = policy_net(state[None, :])
            action_prob = probs[0, action]
            loss -= tf.math.log(action_prob) * return_
        
        gradients = tape.gradient(loss, policy_net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, policy_net.trainable_variables))

# 训练过程
for episode in range(episodes):
    state = env.reset()[0]
    episode_rewards = []
    states, actions, rewards = [], [], []

    done = False
    while not done:
        action, probs = select_action(state)
        next_state, reward, done, truncated, _ = env.step(action)

        # env.render()
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        
        state = next_state
        episode_rewards.append(reward)

    returns = compute_return(rewards, gamma)
    train_step(states, actions, returns)

    if episode % 50 == 0:
        print(f'Episode {episode}, Total Reward: {np.sum(episode_rewards)}')

env.close()
