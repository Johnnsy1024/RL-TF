import random
from loguru import logger
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
from agent import SAC
from slots import *
if __name__ == "__main__":
    agent = SAC(env, state_dim, action_dim, hidden_dim, actor_lr, critic_lr, alpha_lr, gamma, tau, alpha_init, target_entropy, buffer_size)
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc=f'Iteration {i}') as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                state, _ = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    # logger.info(f"action: {action}")
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    agent.memory.append((state, action, reward, next_state, done))
                    state = next_state
                    # logger.info(f"state: {state}")
                    episode_return += reward
                    if len(agent.memory) > batch_size:
                        batch = random.sample(agent.memory, batch_size)
                        b_s, b_a, b_r, b_ns, b_d = zip(*batch)
                        transition_dict = {"states": b_s, "actions": b_a, "rewards": b_r, "next_states": b_ns, "dones": b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    # logger.info(f"current return_list: {return_list}")
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.savefig(f"./prods/sac_{env.spec.name}.png")