import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from agent import TD3
from model import build_actor_model, build_critic_model
from slots import (
    ACTION_DIM,
    ACTOR_LR,
    BATCH_SIZE,
    BUFFER_SIZE,
    CRITIC_LR,
    GAMMA,
    HIDDEN_DIM,
    NOISE_TYPE,
    NUM_EPISODES,
    SIGMA,
    SIGMA_END,
    STATE_DIM,
    TAU,
    env,
)
from tqdm import tqdm
from util import OUActionNoise, ReplayBuffer

if __name__ == "__main__":
    if NOISE_TYPE == "ou":
        noise = OUActionNoise(mean=np.zeros(ACTION_DIM))
    agent = TD3(
        env,
        STATE_DIM,
        ACTION_DIM,
        HIDDEN_DIM,
        ACTOR_LR,
        CRITIC_LR,
        SIGMA,
        SIGMA_END,
        TAU,
        GAMMA,
    )
    buffer = ReplayBuffer()
    return_list = []
    for i in range(10):
        with tqdm(total=int(NUM_EPISODES / 10), desc="Iteration %d" % i) as pbar:
            for i_episode in range(int(NUM_EPISODES / 10)):
                episode_return = 0
                step_count = 0
                state, _ = env.reset()
                if NOISE_TYPE == "ou":
                    noise.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    if NOISE_TYPE == "ou":
                        action += noise()
                    else:
                        action += agent.sigma * np.random.randn(ACTION_DIM)
                        agent.sigma = agent.sigma - (agent.sigma - SIGMA_END) / (
                            NUM_EPISODES // 2
                        )
                    action = np.clip(
                        action, env.action_space.low[0], env.action_space.high[0]
                    )
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    buffer.add(state, action, reward, next_state, done)
                    if len(buffer) > BATCH_SIZE:
                        s, a, r, s2, d = buffer.sample(batch_size=BATCH_SIZE)
                        agent.update(
                            tf.convert_to_tensor(s, dtype=tf.float32),
                            tf.convert_to_tensor(a, dtype=tf.float32),
                            tf.convert_to_tensor(r, dtype=tf.float32),
                            tf.convert_to_tensor(s2, dtype=tf.float32),
                            tf.convert_to_tensor(d, dtype=tf.float32),
                            step_count,
                        )
                        step_count += 1
                    state = next_state
                    episode_return += reward
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
    plt.savefig(f"./prods/ddpg_{env.spec.name}_{NOISE_TYPE}.png")
