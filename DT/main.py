"""
A minimal, reasonably complete Decision Transformer implementation in TensorFlow 2 / Keras.

Features:
- Transformer decoder (causal) that models sequences of (return-to-go, state, action) tokens.
- Supports both continuous actions (MSE) and discrete actions (cross-entropy).
- Utilities to build a tf.data.Dataset from offline trajectories (list of dicts with keys 'obs', 'acts', 'rews').
- Simple training loop using model.fit-compatible Dataset.

Notes / assumptions:
- Trajectories are lists/arrays of length T with:
    * obs: shape (T, obs_dim)
    * acts: shape (T, ) for discrete (int) or (T, act_dim) for continuous
    * rews: shape (T, )
- We form sequences of max length `max_len` (number of timesteps) and interleave tokens: (R_0, s_0, a_0, R_1, s_1, a_1, ...)
- When predicting actions we only compute loss on action-token positions.

Reference: Chen et al., "Decision Transformer" (2021) -- this is a compact implementation for experimentation.

"""

import numpy as np
import tensorflow as tf


# ----------------------------- Transformer blocks -----------------------------
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads
        )
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.ffn = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(d_ff, activation="relu"),
                tf.keras.layers.Dense(d_model),
            ]
        )
        self.dropout2 = tf.keras.layers.Dropout(dropout)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def causal_attention_mask(self, batch_size, seq_len):
        # Lower triangular mask: (seq_len, seq_len)
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        # Keras expects shape [batch, seq_len, seq_len]
        mask = tf.broadcast_to(mask[None, :, :], [batch_size, seq_len, seq_len])
        return mask

    def call(self, x, training=False):
        # x shape: (batch, seq_len, d_model)
        batch = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        attn_mask = self.causal_attention_mask(batch, seq_len)
        attn_output = self.mha(x, x, attention_mask=attn_mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.norm2(out1 + ffn_output)
        return out2


# ----------------------------- Decision Transformer model -----------------------------
class DecisionTransformer(tf.keras.Model):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_len=20,
        d_model=128,
        num_layers=3,
        num_heads=4,
        d_ff=256,
        dropout=0.1,
        discrete_action=False,
        **kwargs,
    ):
        """
        Args:
            state_dim: dimension of state vector
            action_dim: if discrete_action -> number of actions (int). else -> continuous action dim
            max_len: number of timesteps per sequence (not counting token expansion). Actual token seq len = 3*max_len
            discrete_action: bool, whether action space is discrete
        """
        super().__init__(**kwargs)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_len = max_len
        self.d_model = d_model
        self.discrete_action = discrete_action

        # token embedding sizes:
        # - return-to-go is scalar -> embed to d_model
        # - state is vector -> linear projection to d_model
        # - action depends: discrete -> embedding table; continuous -> linear projection
        self.ret_emb = tf.keras.layers.Dense(d_model)
        self.state_emb = tf.keras.layers.Dense(d_model)
        if discrete_action:
            self.act_emb = tf.keras.layers.Embedding(action_dim, d_model)
        else:
            self.act_emb = tf.keras.layers.Dense(d_model)

        # positional embeddings for tokens (length = 3 * max_len)
        self.pos_emb = tf.keras.layers.Embedding(3 * max_len, d_model)

        # transformer stack
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ]
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # output head: when predicting action we map to action_dim
        if discrete_action:
            self.predict_act = tf.keras.layers.Dense(action_dim)  # logits
        else:
            self.predict_act = tf.keras.layers.Dense(action_dim)

    def _embed_sequence(self, returns, states, actions):
        """
        returns: (batch, T) scalar
        states: (batch, T, state_dim)
        actions: (batch, T) either ints (discrete) or floats (continuous actions)

        returns/states/actions correspond to timesteps 0..T-1
        We will interleave token embeddings into sequence length 3T: [r0,s0,a0,r1,s1,a1,...]
        """
        batch = tf.shape(states)[0]
        T = tf.shape(states)[1]

        # embed each type
        ret_e = self.ret_emb(tf.expand_dims(returns, -1))  # (batch, T, d_model)
        state_e = self.state_emb(states)  # (batch, T, d_model)
        if self.discrete_action:
            actions_e = self.act_emb(
                actions
            )  # (batch, T, d_model) actions must be int dtype
        else:
            actions_e = self.act_emb(actions)  # (batch, T, d_model)

        # Now interleave: build a tensor of shape (batch, 3*T, d_model)
        seq = tf.stack([ret_e, state_e, actions_e], axis=2)  # (batch, T, 3, d_model)
        seq = tf.reshape(seq, [batch, 3 * T, self.d_model])

        # add positional embeddings
        positions = tf.range(0, 3 * self.max_len, dtype=tf.int32)[
            None, : tf.shape(seq)[1]
        ]
        pos_e = self.pos_emb(positions)
        seq = seq + pos_e
        return seq

    def call(self, returns, states, actions, training=False):
        """
        returns: (batch, T)
        states: (batch, T, state_dim)
        actions: (batch, T) or (batch, T, action_dim)
        """
        x = self._embed_sequence(returns, states, actions)
        for blk in self.transformer_blocks:
            x = blk(x, training=training)
        x = self.norm(x)
        # x shape (batch, 3T, d_model)
        return x

    def predict_actions_from_transformer_output(self, transformer_output):
        """
        transformer_output: (batch, 3T, d_model)
        We only extract positions corresponding to action tokens: indices 2,5,8,... -> 3*t + 2
        Return logits or continuous predictions of shape (batch, T, action_dim)
        """
        seq_len = tf.shape(transformer_output)[1]
        T = seq_len // 3
        # gather indices
        idxs = tf.range(2, 3 * T, 3)
        action_tokens = tf.gather(transformer_output, idxs, axis=1)
        preds = self.predict_act(action_tokens)
        return preds

    def compute_loss(self, returns, states, actions):
        # forward
        x = self.call(returns, states, actions, training=True)
        preds = self.predict_actions_from_transformer_output(
            x
        )  # (batch, T, action_dim) or logits

        if self.discrete_action:
            # actions are ints of shape (batch, T)
            action_targets = actions
            # compute crossentropy per timestep
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                action_targets, preds, from_logits=True
            )
            return tf.reduce_mean(loss)
        else:
            # continuous: preds are predicted actions (no distribution modeling here)
            action_targets = actions  # shape (batch, T, action_dim)
            loss = tf.keras.losses.mean_squared_error(
                tf.reshape(action_targets, [-1, tf.shape(action_targets)[-1]]),
                tf.reshape(preds, [-1, tf.shape(preds)[-1]]),
            )
            return tf.reduce_mean(loss)


# ----------------------------- Data utilities -----------------------------


def compute_returns_to_go(rews, gamma=1.0):
    # rews: (T,)
    T = len(rews)
    rtg = np.zeros(T, dtype=np.float32)
    running = 0.0
    for t in reversed(range(T)):
        running = rews[t] + gamma * running
        rtg[t] = running
    return rtg


def make_dt_dataset(
    trajectories,
    state_dim,
    action_dim,
    max_len=20,
    batch_size=64,
    discrete_action=False,
    gamma=1.0,
    shuffle=True,
):
    """
    trajectories: list of dicts, each with 'obs' (T, state_dim), 'acts' (T, ) or (T, action_dim), 'rews' (T,)
    Produces a tf.data.Dataset yielding batches:
        returns: (batch, L)
        states: (batch, L, state_dim)
        actions: (batch, L) or (batch, L, action_dim)
    where L = max_len (we pad shorter sequences at the start with zeros)
    """
    seqs = []
    for traj in trajectories:
        obs = np.asarray(traj["obs"], dtype=np.float32)
        acts = np.asarray(traj["acts"])
        rews = np.asarray(traj["rews"], dtype=np.float32)
        T = len(rews)
        rtg = compute_returns_to_go(rews, gamma=gamma)

        # We can create multiple sub-sequences from a long trajectory by sliding window
        # For simplicity, extract windows of length max_len with end aligned to t
        for start in range(0, max(1, T - 0), max_len):
            end = min(start + max_len, T)
            L = end - start
            # pad at the beginning to length max_len
            pad = max_len - L
            r = np.concatenate([np.zeros(pad, dtype=np.float32), rtg[start:end]])
            s = np.concatenate(
                [np.zeros((pad, state_dim), dtype=np.float32), obs[start:end]]
            )
            if discrete_action:
                a = np.concatenate([np.zeros(pad, dtype=np.int32), acts[start:end]])
            else:
                a = np.concatenate(
                    [np.zeros((pad, action_dim), dtype=np.float32), acts[start:end]]
                )
            seqs.append((r, s, a))

    # convert to arrays
    returns = np.stack([s[0] for s in seqs], axis=0)
    states = np.stack([s[1] for s in seqs], axis=0)
    actions = np.stack([s[2] for s in seqs], axis=0)

    ds = tf.data.Dataset.from_tensor_slices((returns, states, actions))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(returns))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ----------------------------- Simple training helper -----------------------------
class DTTrainer:
    def __init__(self, model: DecisionTransformer, lr=1e-4):
        self.model = model
        self.opt = tf.keras.optimizers.Adam(learning_rate=lr)

    @tf.function
    def train_step(self, returns, states, actions):
        with tf.GradientTape() as tape:
            loss = self.model.compute_loss(returns, states, actions)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def fit(self, dataset, epochs=10, steps_per_epoch=None):
        for ep in range(epochs):
            print(f"Epoch {ep+1}/{epochs}")
            for step, (r, s, a) in enumerate(dataset):
                loss = self.train_step(r, s, a)
                if step % 50 == 0:
                    tf.print("step", step, "loss", loss)
                if steps_per_epoch and step >= steps_per_epoch:
                    break


# ----------------------------- Example usage -----------------------------
if __name__ == "__main__":
    # toy example creating random trajectories for testing
    import random

    state_dim = 8
    action_dim = 4
    discrete = True

    # create random trajectories
    trajs = []
    for _ in range(200):
        T = random.randint(10, 60)
        obs = np.random.randn(T, state_dim).astype(np.float32)
        if discrete:
            acts = np.random.randint(0, action_dim, size=(T,))
        else:
            acts = np.random.randn(T, action_dim).astype(np.float32)
        rews = (np.random.randn(T) * 0.1 + 1.0).astype(np.float32)
        trajs.append({"obs": obs, "acts": acts, "rews": rews})

    max_len = 20
    ds = make_dt_dataset(
        trajs,
        state_dim,
        action_dim,
        max_len=max_len,
        batch_size=32,
        discrete_action=discrete,
    )

    model = DecisionTransformer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_len=max_len,
        d_model=128,
        num_layers=3,
        num_heads=4,
        d_ff=256,
        dropout=0.1,
        discrete_action=discrete,
    )

    trainer = DTTrainer(model, lr=1e-4)
    trainer.fit(ds)

    print("Done")
