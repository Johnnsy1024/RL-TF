import numpy as np
import tensorflow as tf
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
                while not done:
                    sample_cnt += 1
                    if epsilon > EPSILON_MIN:
                        epsilon = EPSILON_MIN + (EPSILON_START - EPSILON_MIN) * np.exp(
                            -1.0 * sample_cnt / EPSILON_DECAY
                        )
                    action = agent.get_action(state, epsilon)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    buffer.add(state, action, reward, next_state, done)
                    if len(buffer) > BATCH_SIZE:
                        s, a, r, s2, d = buffer.sample(batch_size=BATCH_SIZE)
                        agent.update(
                            tf.convert_to_tensor(s, dtype=tf.float32),
                            tf.convert_to_tensor(a, dtype=tf.int64),
                            tf.convert_to_tensor(r, dtype=tf.float32),
                            tf.convert_to_tensor(s2, dtype=tf.float32),
                            tf.convert_to_tensor(d, dtype=tf.float32),
                        )
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
