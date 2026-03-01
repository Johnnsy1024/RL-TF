import numpy as np
from agent import REINFORCE
from matplotlib import pyplot as plt
from slots import action_dim, env, gamma, hidden_dim, lr, num_episodes, state_dim
from tqdm import tqdm

if __name__ == "__main__":
    agent = REINFORCE(state_dim, action_dim, hidden_dim, lr, gamma)
    return_list = []

    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc="Iteration %d" % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state, _ = env.reset()
                done = False

                episode_states = []
                episode_actions = []
                episode_rewards = []

                while not done:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    episode_states.append(state)
                    episode_actions.append(action)
                    episode_rewards.append(reward)

                    state = next_state
                    episode_return += reward

                agent.update(episode_states, episode_actions, episode_rewards)
                return_list.append(episode_return)

                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix(
                        {
                            "episode": "%d" % (num_episodes / 10 * i + i_episode + 1),
                            "return": "%.3f" % np.mean(return_list[-10:]),
                        }
                    )
                pbar.update(1)

    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.savefig(f"./prods/reinforce_{env.spec.name}.png")
