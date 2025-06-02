import random

import numpy as np
import tensorflow as tf
from agent import DQN, DuelingDQN
from matplotlib import pyplot as plt
from slots import (
    action_dim,
    batch_size,
    dqn_type,
    env,
    epsilon_decay,
    epsilon_min,
    epsilon_start,
    gamma,
    lr,
    memory_size,
    n_episodes,
    state_dim,
    target_update_freq,
)
from tqdm import tqdm

if __name__ == "__main__":
    if dqn_type == "dqn" or dqn_type == "double_dqn":
        agent = DQN()
    elif dqn_type == "dueling_dqn":
        agent = DuelingDQN()
    return_list = []
    epsilon = epsilon_start
    sample_cnt = 0
    for i in range(10):
        with tqdm(total=int(n_episodes / 10), desc="Iteration %d" % i) as pbar:
            for i_episode in range(int(n_episodes / 10)):
                episode_return = 0
                state = env.reset()[0]
                state = tf.reshape(state, [-1, state_dim])
                done = False
                while not done:
                    sample_cnt += 1
                    if epsilon > epsilon_min:
                        epsilon = epsilon_min + (epsilon_start - epsilon_min) * np.exp(
                            -1.0 * sample_cnt / epsilon_decay
                        )
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
                        actions = np.array(actions).reshape(-1, 1)
                        rewards = np.array(rewards).reshape(-1, 1)
                        dones = np.array(dones).reshape(-1, 1)
                        states = tf.squeeze(tf.stack(states), axis=1)
                        next_states = tf.squeeze(tf.stack(next_states), axis=1)

                        with tf.GradientTape() as tape:
                            q_values = agent.q_network(states, training=True)
                            q_values = tf.gather(q_values, actions, axis=1, batch_dims=1)
                            max_next_q_values = tf.reduce_max(
                                agent.target_network(next_states, training=True),
                                1,
                                keepdims=True,
                            )
                            q_target = rewards + gamma * max_next_q_values * (1 - dones)
                            q_loss = tf.reduce_mean(tf.square(q_target - q_values))
                        grads = tape.gradient(q_loss, agent.q_network.trainable_variables)
                        agent.q_network.optimizer.apply_gradients(
                            zip(grads, agent.q_network.trainable_variables)
                        )

                    agent.update_target_network(sample_cnt)

                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix(
                        {
                            "episode": "%d" % (n_episodes / 10 * i + i_episode + 1),
                            "return": "%.3f" % np.mean(return_list[-10:]),
                        }
                    )
                pbar.update(1)
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.savefig("./prods/dqn_cartpole.png")
