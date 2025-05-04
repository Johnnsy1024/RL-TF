from matplotlib import pyplot as plt
from tqdm import tqdm
from agent import PPO
import tensorflow as tf
import numpy as np 
from slots import env, state_dim, action_dim, actor_lr, critic_lr, num_episodes, hidden_dim, gamma, lmbda, epochs, eps
from loguru import logger

if __name__ == '__main__':
    agent = PPO(env, hidden_dim, state_dim, action_dim, actor_lr, critic_lr, lmbda,
            epochs, eps, gamma)
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()[0]
                state = tf.one_hot(state, depth=state_dim)
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    next_state = tf.one_hot(next_state, depth=state_dim)
                    done = terminated or truncated
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(int(action))
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.savefig(f"./prods/ppo_{env.spec.name}.png")