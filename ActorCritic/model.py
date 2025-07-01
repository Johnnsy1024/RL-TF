import tensorflow as tf
from flags import args


def build_actor_model(action_size: int, inputs: tf.keras.Input):
    x = tf.keras.layers.Dense(64, activation="selu")(inputs)
    x = tf.keras.layers.Dense(action_size, activation="softmax")(x)
    model = tf.keras.Model(inputs, x)
    return model


def build_critic_model(inputs: tf.keras.Input):
    x = tf.keras.layers.Dense(64, activation="selu")(inputs)
    x = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, x)
    return model
