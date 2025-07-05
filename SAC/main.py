import random

import numpy as np
from agent import SAC
from matplotlib import pyplot as plt
from slots import (
    action_dim,
    actor_lr,
    alpha_init,
    alpha_lr,
    batch_size,
    buffer_size,
    critic_lr,
    env,
    env_action_type,
    gamma,
    hidden_dim,
    num_episodes,
    state_dim,
    target_entropy,
    tau,
)
from tqdm import tqdm

if __name__ == "__main__":
    agent = SAC(
        env,
        state_dim,
        action_dim,
        hidden_dim,
        actor_lr,
        critic_lr,
        alpha_lr,
        gamma,
        tau,
        alpha_init,
        target_entropy if env_action_type == "continuous" else -1,
        buffer_size,
    )
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc=f"Iteration {i}") as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state, _ = env.reset()
                done = False
                while not done:
                    action = (
                        agent.take_action(state)
                        if env_action_type == "continuous"
                        else agent.take_action_discrete(state)
                    )
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    agent.memory.append((state, action, reward, next_state, done))
                    state = next_state
                    episode_return += reward
                    if len(agent.memory) > batch_size:
                        batch = random.sample(agent.memory, batch_size)
                        b_s, b_a, b_r, b_ns, b_d = map(np.array, zip(*batch))
                        b_r = b_r.reshape(-1, 1)
                        b_d = b_d.reshape(-1, 1)
                        transition_dict = {
                            "states": b_s,
                            "actions": b_a,
                            "rewards": b_r,
                            "next_states": b_ns,
                            "dones": b_d,
                        }
                        (
                            agent.update(transition_dict)
                            if env_action_type == "continuous"
                            else agent.update_discrete(transition_dict)
                        )
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    # logger.info(f"current return_list: {return_list}")
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
    plt.savefig(f"./prods/sac_{env.spec.name}.png")
