import tensorflow as tf


def build_actor_model(hidden_dim: int, action_dim: int, actor_input: tf.keras.Input):
    x = tf.keras.layers.Dense(hidden_dim, activation='relu')(actor_input)
    x = tf.keras.layers.Dense(action_dim, activation='softmax')(x)
    actor_model = tf.keras.Model(actor_input, x)
    return actor_model

def build_critic_model(hidden_dim: int, critic_input: tf.keras.Input):
    x = tf.keras.layers.Dense(hidden_dim, activation='relu')(critic_input)
    x = tf.keras.layers.Dense(1)(x)
    critic_model = tf.keras.Model(critic_input, x)
    return critic_model

if __name__ == "__main__":
    actor = build_actor_model(128, 2, tf.keras.Input(shape=(10,)))
    print(actor.summary())