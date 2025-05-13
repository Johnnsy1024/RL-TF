import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def build_actor(state_dim: int, hidden_dim: int, action_dim: int, actor_type: str, action_bound: float):
    if actor_type == "continuous":
        actor_input = tf.keras.layers.Input(shape=(state_dim,), name="actor_input")
        x = tf.keras.layers.Dense(hidden_dim, activation="relu")(actor_input)
        mu = tf.keras.layers.Dense(action_dim, name="mu")(x)
        std = tf.keras.layers.Dense(action_dim, name="std", activation="softplus")(x)
        dist = tfd.Normal(mu, std)
        normal_sample = dist.sample() # 直接重参数化采样前驱变量
        action = tf.tanh(normal_sample)
        log_prob = dist.log_prob(normal_sample) - tf.math.log(1 - tf.tanh(action).pow(2) + 1e-7)
        action = action * action_bound
        actor_model = tf.keras.Model(actor_input, [action, log_prob])
        return actor_model
    elif actor_type == "discrete":
        pass
    
def build_critic(state_dim: int, hidden_dim: int, action_dim: int):
    critic_input_state = tf.keras.layers.Input(shape=(state_dim,), name="critic_input_state")
    critic_input_action = tf.keras.layers.Input(shape=(action_dim,), name="critic_input_action")
    x = tf.keras.layers.Concatenate()([critic_input_state, critic_input_action])
    x = tf.keras.layers.Dense(hidden_dim, activation="relu")(x)
    x = tf.keras.layers.Dense(hidden_dim, activation="relu")(x)
    x = tf.keras.layers.Dense(1, name="q_value")(x)
    critic_model = tf.keras.Model(inputs=[critic_input_state, critic_input_action], outputs=x)
    return critic_model