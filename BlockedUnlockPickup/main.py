from re import L

import numpy as np
from agent import DQN, DuelingDQN
from matplotlib import pyplot as plt
from slots import (
    BATCH_SIZE,
    BUFFER_SIZE,
    DQN_TYPE,
    EPSILON_DECAY,
    EPSILON_MIN,
    EPSILON_START,
    NUM_EPISODES,
    env,
)
from tqdm import tqdm
from util import ReplayBuffer

if __name__ == "__main__":
    if DQN_TYPE == "dqn" or DQN_TYPE == "double_dqn":
        agent = DQN()
    elif DQN_TYPE == "dueling_dqn":
        agent = DuelingDQN()
    buffer = ReplayBuffer(BUFFER_SIZE)
    return_list = []
    epsilon = EPSILON_START
    sample_cnt = 0
    for i in range(10):
        with tqdm(total=int(NUM_EPISODES / 10), desc="Iteration %d" % i) as pbar:
            for i_episode in range(int(NUM_EPISODES / 10)):
                episode_return = 0
                state, _ = env.reset()
                done = False
                state["image"] = state["image"][None, :, :, :]
                state["direction"] = np.array(state["direction"])[None][None, :]
                del state["mission"]
                while not done:
                    sample_cnt += 1
                    if epsilon > EPSILON_MIN:
                        epsilon = EPSILON_MIN + (EPSILON_START - EPSILON_MIN) * np.exp(
                            -1.0 * sample_cnt / EPSILON_DECAY
                        )
                    action = agent.get_action(
                        state,
                        epsilon,
                    )
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    next_state["image"] = next_state["image"][None, :, :, :]
                    next_state["direction"] = np.array(next_state["direction"])[None][
                        None, :
                    ]
                    del next_state["mission"]
                    done = terminated or truncated
                    buffer.add(state, action, reward, next_state, done)
                    if len(buffer) > BATCH_SIZE:
                        s, a, r, s2, d = buffer.sample(batch_size=BATCH_SIZE)
                        s = {
                            "image": np.stack([item["image"] for item in s]).squeeze(),
                            "direction": np.array(
                                [item["direction"] for item in s]
                            ).reshape(-1, 1),
                        }
                        s2 = {
                            "image": np.stack([item["image"] for item in s2]).squeeze(),
                            "direction": np.array(
                                [item["direction"] for item in s2]
                            ).reshape(-1, 1),
                        }

                        agent.update(s, a, r, s2, d)
                    state = next_state
                    episode_return += reward
                    agent.update_target_network(sample_cnt)

                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix(
                        {
                            "episode": "%d" % (NUM_EPISODES / 10 * i + i_episode + 1),
                            "return": "%.3f" % np.mean(return_list[-10:]),
                        }
                    )
                pbar.update(1)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.savefig(f"./prods/dqn_{env.spec.name}_{DQN_TYPE}.png")
