import tensorflow as tf


def build_policy_model(hidden_dim: int, action_dim: int, policy_input: tf.keras.Input):
    x = tf.keras.layers.Dense(hidden_dim, activation="relu")(policy_input)
    x = tf.keras.layers.Dense(action_dim, activation="softmax")(x)
    policy_model = tf.keras.Model(policy_input, x)
    return policy_model


if __name__ == "__main__":
    policy = build_policy_model(128, 2, tf.keras.Input(shape=(4,)))
    print(policy.summary())
