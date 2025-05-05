import tensorflow as tf


def build_actor_model(action_dim: int, hidden_dim: int, action_bound: float, actor_input: tf.keras.Input):
    x = tf.keras.layers.Dense(hidden_dim, activation="relu")(actor_input)
    x = tf.keras.layers.Dense(action_dim, activation="tanh")(x) * action_bound
    return tf.keras.Model(inputs=actor_input, outputs=x)
    
def build_critic_model(hidden_dim: int, critic_input_state: tf.keras.Input, critic_input_action: tf.keras.Input):
    
    x = tf.keras.layers.Concatenate()([critic_input_state, critic_input_action])
    x = tf.keras.layers.Dense(hidden_dim, activation="relu")(x)
    x = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs=[critic_input_state, critic_input_action], outputs=x)