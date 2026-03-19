import tensorflow as tf
from slots import HIDDEN_DIM, IMAGE_EMBED_SIZE, LR, STATE_DIR_CNT, STATE_IMG_SHAPE


def build_model(action_size: int):
    image_inputs = tf.keras.Input(shape=STATE_IMG_SHAPE, name="image", dtype="int32")
    direction_inputs = tf.keras.Input(
        shape=(1,), name="direction", dtype="int32"
    )  # 修正方向输入维度

    # 直接使用切片获取三个通道，形状变为 (None, 7, 7)
    obj_in = image_inputs[:, :, :, 0]
    clr_in = image_inputs[:, :, :, 1]
    sta_in = image_inputs[:, :, :, 2]

    # Embedding 处理
    # 注意：不要先 Flatten 再 Concatenate，这样会丢失空间特征
    # 建议先 Embedding -> Concatenate -> Conv2D/Flatten
    obj_embed = tf.keras.layers.Embedding(IMAGE_EMBED_SIZE, HIDDEN_DIM)(obj_in)
    clr_embed = tf.keras.layers.Embedding(IMAGE_EMBED_SIZE, HIDDEN_DIM)(clr_in)
    sta_embed = tf.keras.layers.Embedding(IMAGE_EMBED_SIZE, HIDDEN_DIM)(sta_in)

    image_embed = tf.keras.layers.Concatenate(axis=-1)([obj_embed, clr_embed, sta_embed])
    image_flat = tf.keras.layers.Flatten()(image_embed)

    # 方向处理
    # MiniGrid 的 direction 通常是一个 0-3 的整数，用 Embedding 很好
    dir_embed = tf.keras.layers.Embedding(STATE_DIR_CNT, HIDDEN_DIM)(direction_inputs)
    dir_flat = tf.keras.layers.Flatten()(dir_embed)

    # 合并
    total_input = tf.keras.layers.Concatenate()([image_flat, dir_flat])

    x = tf.keras.layers.Dense(HIDDEN_DIM, activation="relu")(total_input)
    x = tf.keras.layers.Dense(HIDDEN_DIM, activation="relu")(x)
    x = tf.keras.layers.Dense(HIDDEN_DIM, activation="relu")(x)
    x = tf.keras.layers.Dense(HIDDEN_DIM, activation="relu")(x)
    x = tf.keras.layers.Dense(action_size)(x)
    model = tf.keras.Model({"image": image_inputs, "direction": direction_inputs}, x)
    model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="mse")
    return model


def build_dueling_model(action_size: int):
    image_inputs = tf.keras.Input(shape=STATE_IMG_SHAPE, name="image", dtype="int32")
    direction_inputs = tf.keras.Input(
        shape=(1,), name="direction", dtype="int32"
    )  # 修正方向输入维度

    # 直接使用切片获取三个通道，形状变为 (None, 7, 7)
    obj_in = image_inputs[:, :, :, 0]
    clr_in = image_inputs[:, :, :, 1]
    sta_in = image_inputs[:, :, :, 2]

    # Embedding 处理
    # 注意：不要先 Flatten 再 Concatenate，这样会丢失空间特征
    # 建议先 Embedding -> Concatenate -> Conv2D/Flatten
    obj_embed = tf.keras.layers.Embedding(IMAGE_EMBED_SIZE, HIDDEN_DIM)(obj_in)
    clr_embed = tf.keras.layers.Embedding(IMAGE_EMBED_SIZE, HIDDEN_DIM)(clr_in)
    sta_embed = tf.keras.layers.Embedding(IMAGE_EMBED_SIZE, HIDDEN_DIM)(sta_in)

    image_embed = tf.keras.layers.Concatenate(axis=-1)([obj_embed, clr_embed, sta_embed])
    image_flat = tf.keras.layers.Flatten()(image_embed)

    # 方向处理
    # MiniGrid 的 direction 通常是一个 0-3 的整数，用 Embedding 很好
    dir_embed = tf.keras.layers.Embedding(STATE_DIR_CNT, HIDDEN_DIM)(direction_inputs)
    dir_flat = tf.keras.layers.Flatten()(dir_embed)

    # 合并
    total_input = tf.keras.layers.Concatenate()([image_flat, dir_flat])

    x = tf.keras.layers.Dense(HIDDEN_DIM, activation="relu")(total_input)
    x = tf.keras.layers.Dense(HIDDEN_DIM, activation="relu")(x)
    x = tf.keras.layers.Dense(HIDDEN_DIM, activation="relu")(x)
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
