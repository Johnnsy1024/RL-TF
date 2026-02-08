import tensorflow as tf
from slots import HIDDEN_DIM, LR, STATE_DIR_CNT, STATE_DIR_DIM, STATE_IMG_SHAPE


def build_model(action_size: int):
    image_inputs = tf.keras.Input(shape=STATE_IMG_SHAPE)
    direction_inputs = tf.keras.Input(shape=(STATE_DIR_DIM,))

    image = tf.keras.layers.Conv2D(32, (2, 2), strides=(1, 1), activation="relu")(
        image_inputs
    )
    image = tf.keras.layers.Conv2D(64, (2, 2), strides=(1, 1), activation="relu")(image)
    image = tf.keras.layers.GlobalAveragePooling2D()(image)

    direction = tf.keras.layers.Embedding(input_dim=STATE_DIR_CNT, output_dim=HIDDEN_DIM)(
        direction_inputs
    )
    direction = tf.keras.layers.Flatten()(direction)
    direction = tf.keras.layers.Dense(HIDDEN_DIM, activation="relu")(direction)
    inputs = tf.keras.layers.Concatenate()([image, direction])
    x = tf.keras.layers.Dense(HIDDEN_DIM, activation="relu")(inputs)
    x = tf.keras.layers.Dense(HIDDEN_DIM, activation="relu")(x)
    x = tf.keras.layers.Dense(action_size)(x)
    model = tf.keras.Model({"image": image_inputs, "direction": direction_inputs}, x)
    model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="mse")
    return model


def build_dueling_model(action_size: int):
    image_inputs = tf.keras.Input(shape=STATE_IMG_SHAPE)
    direction_inputs = tf.keras.Input(shape=(STATE_DIR_DIM,))

    image = tf.keras.layers.Conv2D(32, (2, 2), strides=(1, 1), activation="relu")(
        image_inputs
    )
    image = tf.keras.layers.Conv2D(64, (2, 2), strides=(1, 1), activation="relu")(image)
    image = tf.keras.layers.GlobalAveragePooling2D()(image)

    direction = tf.keras.layers.Embedding(input_dim=STATE_DIR_CNT, output_dim=HIDDEN_DIM)(
        direction_inputs
    )
    direction = tf.keras.layers.Flatten()(direction)
    direction = tf.keras.layers.Dense(HIDDEN_DIM, activation="relu")(direction)
    inputs = tf.keras.layers.Concatenate()([image, direction])

    x = tf.keras.layers.Dense(HIDDEN_DIM, activation="relu")(inputs)
    x = tf.keras.layers.Dense(HIDDEN_DIM, activation="relu")(x)
    x_v = tf.keras.layers.Dense(1)(x)
    x_a = tf.keras.layers.Dense(action_size)(x)
    mean_a = tf.reduce_mean(x_a, axis=-1, keepdims=True)
    x_a_adj = tf.keras.layers.Subtract()([x_a, mean_a])
    q_values = tf.keras.layers.Add()([x_v, x_a_adj])
    model = tf.keras.Model(
        {"image": image_inputs, "direction": direction_inputs}, q_values
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="mse")
    return model


if __name__ == "__main__":
    x = tf.convert_to_tensor([[1, 2, 3]])
    inputs = tf.keras.Input(shape=(3,))
    model = build_model(2, inputs)
    print(model.summary())
    print(model(x))
