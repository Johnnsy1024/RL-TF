import tensorflow as tf
import tensorflow_probability as tfp
from matplotlib import scale

tfd = tfp.distributions


def build_actor(
    state_dim: int,
    hidden_dim: int,
    action_dim: int,
    action_bound: float,
    actor_type: str = "continuous",
):
    if actor_type == "continuous":
        actor_input = tf.keras.layers.Input(shape=(state_dim,), name="actor_input")
        x = tf.keras.layers.Dense(hidden_dim, activation="relu")(actor_input)
        mu = tf.keras.layers.Dense(action_dim, name="mu")(x)
        std = tf.keras.layers.Dense(action_dim, name="std", activation="softplus")(x)
        min_std = 1e-4
        std = tf.clip_by_value(std, min_std, 1)

        def sample_action(args):
            mu, std = args
            dist = tfd.Normal(loc=mu, scale=std)
            normal_sample = dist.sample()
            action = tf.tanh(normal_sample)
            log_prob = dist.log_prob(normal_sample)
            log_prob = tf.reduce_sum(log_prob, axis=-1, keepdims=True)
            log_prob -= tf.reduce_sum(
                tf.math.log(1 - tf.square(tf.tanh(normal_sample)) + 1e-7),
                axis=-1,
                keepdims=True,
            )
            action = action * action_bound
            return action, log_prob

        action, log_prob = tf.keras.layers.Lambda(sample_action)([mu, std])
        actor_model = tf.keras.Model(actor_input, [action, log_prob])
        return actor_model
    elif actor_type == "discrete":
        pass


def build_critic(state_dim: int, hidden_dim: int, action_dim: int):
    critic_input_state = tf.keras.layers.Input(
        shape=(state_dim,), name="critic_input_state"
    )
    critic_input_action = tf.keras.layers.Input(
        shape=(action_dim,), name="critic_input_action"
    )
    x = tf.keras.layers.Concatenate()([critic_input_state, critic_input_action])
    x = tf.keras.layers.Dense(hidden_dim, activation="relu")(x)
    x = tf.keras.layers.Dense(hidden_dim, activation="relu")(x)
    x = tf.keras.layers.Dense(1, name="q_value")(x)
    critic_model = tf.keras.Model(
        inputs=[critic_input_state, critic_input_action], outputs=x
    )
    return critic_model
