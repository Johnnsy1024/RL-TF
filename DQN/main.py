import random
from matplotlib import pyplot as plt
import tensorflow as tf
from agent import DQN, DuelingDQN
from tqdm import tqdm
from slots import env, lr, memory_size, epsilon, epsilon_min, epsilon_decay, batch_size, target_update_freq, n_episodes, state_dim, action_dim, gamma, dqn_type
import numpy as np

if __name__ == "__main__":
    if dqn_type == 'dqn' or dqn_type == 'double_dqn':
        agent = DQN()
    elif dqn_type == 'dueling_dqn':
        agent = DuelingDQN()
    return_list = []
    for i in range(10):
        with tqdm(total=int(n_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(n_episodes/10)):
                episode_return = 0
                state = env.reset()[0]
                state = tf.reshape(state, [-1, state_dim])
                done = False
                while not done:
                    action = agent.get_action(state, epsilon)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    next_state = tf.reshape(next_state, [-1, state_dim])
                    
                    agent.memory.append((state, action, reward, next_state, done))
                    state = next_state
                    episode_return += reward
                    
                    if len(agent.memory) >= batch_size:
                        batch = random.sample(agent.memory, batch_size)
                        states, actions, rewards, next_states, dones = zip(*batch)
                        states = tf.squeeze(tf.stack(states), axis=1)
                        next_states = tf.squeeze(tf.stack(next_states), axis=1)
                        q_values = agent.q_network.predict(states, verbose=0)
                        next_q_values = agent.target_network.predict(next_states, verbose=0)
                        
                        for b in range(batch_size):
                            target = rewards[b]
                            if not dones[b]:
                                if dqn_type == 'double_dqn':
                                    # 主网络选动作+目标网络估值
                                    best_action = np.argmax(agent.q_network.predict(next_states[b], verbose=0)[b])
                                    target += gamma * next_q_values[b][best_action]
                                else:
                                    target += gamma * np.max(next_q_values[b])
                            q_values[b][actions[b]] = target
                        agent.q_network.fit(states, q_values, epochs=1, verbose=0)
                agent.update_target_network(i_episode)
                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (n_episodes/10 * i + i_episode + 1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.savefig("./prods/dqn_cartpole.png")