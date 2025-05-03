import tensorflow as tf
from flags import args

def build_model(action_size: int, inputs: tf.keras.Input):
    x = tf.keras.layers.Dense(64, activation="selu")(inputs)
    x = tf.keras.layers.Dense(action_size)(x)
    model = tf.keras.Model(inputs, x)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss='mse')
    return model

def build_dueling_model(action_size: int, inputs: tf.keras.Input):
    x = tf.keras.layers.Dense(64, activation="selu")(inputs)
    x_v = tf.keras.layers.Dense(1)(x)
    x_a = tf.keras.layers.Dense(action_size)(x)
    mean_a = tf.reduce_mean(x_a, axis=-1, keepdims=True)
    x_a_adj = tf.keras.layers.Subtract()([x_a, mean_a])
    q_values = tf.keras.layers.Add()([x_v, x_a_adj])
    model = tf.keras.Model(inputs, q_values)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss='mse')
    return model

if __name__ == "__main__":
    x = tf.convert_to_tensor([[1, 2, 3]])
    inputs = tf.keras.Input(shape=(3,))
    model = build_model(2, inputs)
    print(model.summary())
    print(model(x))