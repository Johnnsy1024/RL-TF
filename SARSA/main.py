from matplotlib import pyplot as plt
from tqdm import tqdm
from agent import SARSA
import numpy as np 
from slots import env, epsilon_end, epsilon_decay, epsilon_start, n_episodes, alpha, gamma, method

if __name__ == '__main__':
    agent = SARSA(env, epsilon_end, epsilon_start, epsilon_decay, alpha, gamma)
    return_list = []
    for i in range(10):
        with tqdm(total=int(n_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(n_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()[0]
                done = False
                action = agent.sample_action(state)
                while not done:
                    next_state, reward, done, trun, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    next_action = agent.sample_action(next_state)
                    agent.update(state, action, reward, next_state, done) if method != "sarsa" else agent.update_sarsa(state, action, reward, next_state, next_action, done)
                    state = next_state
                    action = next_action
                    episode_return += reward
                return_list.append(episode_return)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (n_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    if method == "sarsa":
        plt.savefig("./prods/sarsa_cliffwalking.png")
    else:
        plt.savefig("./prods/qlearning_cliffwalking.png")