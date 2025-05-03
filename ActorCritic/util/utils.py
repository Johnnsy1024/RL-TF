import numpy as np
import tensorflow as tf

def compute_return(rewards, gamma):
    returns = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = rewards[t] + gamma * running_add
        returns[t] = running_add
    return returns

def train_step(states, actions, returns, advantages, actor_net, critic_net, actor_optimizer, critic_optimizer):
    with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
        actor_loss = 0
        critic_loss = 0
        for i in range(len(states)):
            state = tf.reshape(tf.convert_to_tensor(states[i], dtype=tf.float32), [-1, states[i].shape[-1]])
            action = actions[i]
            return_ = returns[i]
            advantage = advantages[i]

            # 计算Actor损失
            probs = actor_net(state)
            action_prob = probs[0, action]
            actor_loss -= tf.math.log(action_prob) * advantage

            # 计算Critic损失
            value = critic_net(state)
            critic_loss += tf.square(return_ - value)

        # 更新网络
        actor_gradients = actor_tape.gradient(actor_loss, actor_net.trainable_variables)
        critic_gradients = critic_tape.gradient(
            critic_loss, critic_net.trainable_variables
        )

        actor_optimizer.apply_gradients(
            zip(actor_gradients, actor_net.trainable_variables)
        )
        critic_optimizer.apply_gradients(
            zip(critic_gradients, critic_net.trainable_variables)
        )