import tensorflow as tf


def build_actor_model(
    state_dim: int, action_dim: int, hidden_dim: int, action_bound: float
):
    inputs = tf.keras.layers.Input(shape=(state_dim,), name="actor_input: state")
    x = tf.keras.layers.Dense(hidden_dim, activation="relu")(inputs)
    x = tf.keras.layers.Dense(hidden_dim, activation="relu")(x)
    output = tf.keras.layers.Dense(action_dim, activation="tanh")(x)
    output = output * action_bound
    return tf.keras.Model(inputs=inputs, outputs=output)


def build_critic_model(state_dim: int, action_dim: int, hidden_dim: int):
    state_input = tf.keras.layers.Input(shape=(state_dim,), name="critic_input: state")
    action_input = tf.keras.layers.Input(shape=(action_dim,), name="critic_input: action")
    concat = tf.keras.layers.Concatenate()([state_input, action_input])
    x = tf.keras.layers.Dense(hidden_dim, activation="relu")(concat)
    x = tf.keras.layers.Dense(hidden_dim, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs=[state_input, action_input], outputs=outputs)
