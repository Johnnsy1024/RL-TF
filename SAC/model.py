import tensorflow as tf


def build_actor(
    state_dim: int,
    hidden_dim: int,
    action_dim: int,
    actor_type: str = "continuous",
):
    actor_input = tf.keras.layers.Input(shape=(state_dim,), name="actor_input")
    x = tf.keras.layers.Dense(hidden_dim, activation="relu")(actor_input)
    x = tf.keras.layers.Dense(hidden_dim, activation="relu")(x)
    if actor_type == "continuous":

        mean = tf.keras.layers.Dense(action_dim, name="mu")(x)
        log_std = tf.keras.layers.Dense(action_dim, name="std")(x)

        actor_model = tf.keras.Model(actor_input, [mean, log_std])
        return actor_model
    elif actor_type == "discrete":
        logits = tf.keras.layers.Dense(action_dim, name="logits")(x)
        actor_model = tf.keras.Model(actor_input, logits)
        return actor_model


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
