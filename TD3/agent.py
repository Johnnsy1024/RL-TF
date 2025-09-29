import gym
import tensorflow as tf
from model import build_actor_model, build_critic_model
from slots import ACTION_DIM, BATCH_SIZE, NOISE_CLIP, POLICY_DELAY, POLICY_NOISE


class TD3:
    def __init__(
        self,
        env: gym.Env,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        actor_lr: float,
        critic_lr: float,
        sigma: float,
        sigma_end: float,
        tau: float,
        gamma: float,
        noise_type: str = "normal",
    ):
        if noise_type not in ["normal", "ou"]:
            raise ValueError("noise_type must be 'normal' or 'ou'")
        self.noise_type = noise_type
        self.env = env
        self.env_name = env.spec.name

        self.action_bound = env.action_space.high[0]
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actor = build_actor_model(
            state_dim, action_dim, hidden_dim, self.action_bound
        )
        self.actor_target = build_actor_model(
            state_dim, action_dim, hidden_dim, self.action_bound
        )
        self.critic_1 = build_critic_model(state_dim, action_dim, hidden_dim)
        self.critic_2 = build_critic_model(state_dim, action_dim, hidden_dim)
        self.critic_target_1 = build_critic_model(state_dim, action_dim, hidden_dim)
        self.critic_target_2 = build_critic_model(state_dim, action_dim, hidden_dim)

        self.actor_target.set_weights(self.actor.get_weights())
        self.critic_target_1.set_weights(self.critic_1.get_weights())
        self.critic_target_2.set_weights(self.critic_2.get_weights())

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer_1 = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.critic_optimizer_2 = tf.keras.optimizers.Adam(learning_rate=critic_lr)

        self.sigma = sigma
        self.sigma_end = sigma_end
        self.tau = tau
        self.gamma = gamma

    def take_action(self, state):
        state = tf.convert_to_tensor(state.reshape(1, -1), dtype=tf.float32)
        action = self.actor(state)[0].numpy()
        return action

    @tf.function
    def update(self, state, action, reward, next_state, done, step_count):
        noise_target = tf.clip_by_value(
            tf.random.normal(shape=(BATCH_SIZE, ACTION_DIM), stddev=POLICY_NOISE),
            -NOISE_CLIP,
            NOISE_CLIP,
        )
        next_action = self.actor_target(next_state) + noise_target
        next_action = tf.clip_by_value(next_action, -self.action_bound, self.action_bound)

        target_q1 = self.critic_target_1([next_state, next_action])
        target_q2 = self.critic_target_2([next_state, next_action])
        target_q = tf.minimum(target_q1, target_q2)
        target_q = reward + self.gamma * (1 - done) * target_q

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            q1 = self.critic_1([state, action])
            q2 = self.critic_2([state, action])
            critic_loss1 = tf.reduce_mean(tf.square(target_q - q1))
            critic_loss2 = tf.reduce_mean(tf.square(target_q - q2))
        grads1 = tape1.gradient(critic_loss1, self.critic_1.trainable_variables)
        grads2 = tape2.gradient(critic_loss2, self.critic_2.trainable_variables)
        self.critic_optimizer_1.apply_gradients(
            zip(grads1, self.critic_1.trainable_variables)
        )
        self.critic_optimizer_2.apply_gradients(
            zip(grads2, self.critic_2.trainable_variables)
        )

        if step_count % POLICY_DELAY == 0:
            with tf.GradientTape() as tape:
                actions = self.actor(state)
                actor_loss = -tf.reduce_mean(self.critic_1([state, actions]))
            grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(
                zip(grads, self.actor.trainable_variables)
            )

            for a, ta in zip(self.actor.variables, self.actor_target.variables):
                ta.assign(self.tau * a + (1 - self.tau) * ta)
            for c1, tc1 in zip(self.critic_1.variables, self.critic_target_1.variables):
                tc1.assign(self.tau * c1 + (1 - self.tau) * tc1)
            for c2, tc2 in zip(self.critic_2.variables, self.critic_target_2.variables):
                tc2.assign(self.tau * c2 + (1 - self.tau) * tc2)
