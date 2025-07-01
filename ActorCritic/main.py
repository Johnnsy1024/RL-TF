from util.utils import compute_return, train_step
from matplotlib import pyplot as plt
import tensorflow as tf
from agent import ActorCritic
from tqdm import tqdm
from slots import env, n_episodes, state_dim, gamma, lr
import numpy as np

if __name__ == "__main__":
    agent = ActorCritic()
    return_list = []
    actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    for i in range(10):
        with tqdm(total=int(n_episodes / 10), desc="Iteration %d" % i) as pbar:
            for i_episode in range(int(n_episodes / 10)):
                episode_rewards = []
                state = env.reset()[0]
                states, actions, rewards, values = [], [], [], []
                done = False
                while not done:
                    action = agent.get_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    value = agent.critic_model(
                        tf.reshape(
                            tf.convert_to_tensor(state, dtype=tf.float32),
                            [-1, state_dim],
                        )
                    )
                    values.append(value.numpy())
                    state = next_state
                    episode_rewards.append(reward)
                returns = compute_return(rewards, gamma)
                advantages = returns - np.array(values).flatten()
                train_step(
                    states,
                    actions,
                    returns,
                    advantages,
                    agent.actor_model,
                    agent.critic_model,
                    actor_optimizer,
                    critic_optimizer,
                )
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix(
                        {
                            "episode": "%d" % (n_episodes / 10 * i + i_episode + 1),
                            "return": "%.3f" % np.sum(episode_rewards),
                        }
                    )
                pbar.update(1)
                return_list.append(np.sum(episode_rewards))
    episodes_list = list(range(len(return_list)))
    plt.plot(episodes_list, return_list)
    plt.xlabel("Episodes")
    plt.ylabel("Returns")
    plt.savefig("./prods/actor_critic_cartpole.png")
