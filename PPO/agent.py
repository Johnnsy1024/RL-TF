import tensorflow as tf
import gym
from model import build_actor_model, build_critic_model

class PPO:
    ''' PPO算法,采用截断方式 '''
    def __init__(self, env: gym.Env, hidden_dim: int, action_dim: int, actor_lr: float, critic_lr: float, lmbda: float, epochs: int, eps: float, gamma: float):
        env_name = env.spec.id
        if env_name != "CliffWalking-v0":
            raise ValueError("Unsupported environment: {}".format(env_name))
        self.action_dim = env.action_space.n
        self.state_dim = env.observation_space.n
        actor_input = tf.keras.Input(shape=(self.state_dim,), name='actor_input', dtype=tf.float32)
        critic_input = tf.keras.Input(shape=(self.state_dim,), name='critic_input', dtype=tf.float32)
        self.actor = build_actor_model(hidden_dim, action_dim, actor_input)
        self.critic = build_critic_model(hidden_dim, critic_input)

        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)

        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数

    def take_action(self, state):
        state = tf.reshape(state, [-1, self.state_dim])
        probs = self.actor(state)
        action = tf.random.categorical(probs, 1).numpy()
        # action_dist = torch.distributions.Categorical(probs) # 构建离散分布
        # action = action_dist.sample()
        return action

    def compute_advantage(self, gamma: float, lmbda: float, td_delta):
        td_delta = td_delta.numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return tf.convert_to_tensor(advantage_list, dtype=tf.float32)
    
    def update(self, transition_dict):
        states = tf.convert_to_tensor(transition_dict['states'], dtype=tf.int32)
        states = tf.one_hot(states, depth=self.state_dim, dtype=tf.int32)
        actions = tf.convert_to_tensor(transition_dict['actions'], dtype=tf.int32)[:, None]
        
        rewards = tf.convert_to_tensor(transition_dict['rewards'], dtype=tf.float32)[:, None]
        next_states = tf.convert_to_tensor(transition_dict['next_states'], dtype=tf.int32)
        next_states = tf.one_hot(next_states, depth=self.state_dim, dtype=tf.int32)
        dones = tf.convert_to_tensor(transition_dict['dones'], dtype=tf.float32)[:, None]
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = self.compute_advantage(self.gamma, self.lmbda, td_delta)
        old_log_probs = tf.gather(self.actor(states), actions, axis=1, batch_dims=1).numpy()

        for _ in range(self.epochs):
            with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
                logits = self.actor(states)
                probs = tf.gather(logits, actions, axis=1, batch_dims=1)
                log_prbos = tf.math.log(probs + 1e-8)
                ratio = tf.math.exp(log_prbos - old_log_probs)
                surr1 = ratio * advantage
                surr2 = tf.clip_by_value(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
                actor_loss = -tf.math.reduce_mean(tf.math.minimum(surr1, surr2))  # PPO损失函数
                critic_loss = tf.math.reduce_mean(tf.math.square(self.critic(states) - td_target))
            actor_grads = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
            critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
            # log_probs = tf.math.log(self.actor(states).gather(1, actions))
            # ratio = tf.math.exp(log_probs - old_log_probs)
            # surr1 = ratio * advantage
            # surr2 = tf.clip_by_value(ratio, 1 - self.eps, 1 + self.eps) * advantage  # 截断
            # actor_loss = tf.math.reduce_mean(-tf.math.minimum(surr1, surr2))  # PPO损失函数
            # critic_loss = tf.math.reduce_mean(tf.math.square(self.critic(states) - td_target))
            # self.actor_optimizer.minimize(actor_loss)
            # self.critic_optimizer.minimize(critic_loss)
            # self.actor_optimizer.step()
            # self.critic_optimizer.step()