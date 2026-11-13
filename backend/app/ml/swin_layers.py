"""Custom layers required to deserialize the supplied Swin Transformer artifact.

These layers mirror the architecture recorded in ``swin_transformer.keras``.  They
are inference-only; no training behaviour or model weights are defined here.
"""
from __future__ import annotations

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="NeuroLens")
class PatchEmbedding(tf.keras.layers.Layer):
    def __init__(self, patch_size=4, embed_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.projection = tf.keras.layers.Conv2D(embed_dim, patch_size, strides=patch_size, name="projection")

    def build(self, input_shape):
        self.built = True

    def call(self, inputs):
        x = self.projection(inputs)
        return tf.reshape(x, [tf.shape(x)[0], -1, self.embed_dim])

    def get_config(self):
        return {**super().get_config(), "patch_size": self.patch_size, "embed_dim": self.embed_dim}


@tf.keras.utils.register_keras_serializable(package="NeuroLens")
class PatchMerging(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim, self.output_dim = input_dim, output_dim
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm")
        self.reduction = tf.keras.layers.Dense(output_dim, use_bias=False, name="reduction")

    def build(self, input_shape):
        self.built = True

    def call(self, x):
        x0, x1 = x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :]
        x2, x3 = x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]
        x = tf.concat([x0, x1, x2, x3], axis=-1)
        return self.reduction(self.norm(x))

    def get_config(self):
        return {**super().get_config(), "input_dim": self.input_dim, "output_dim": self.output_dim}


class WindowAttention(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.dim, self.num_heads = dim, num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = tf.keras.layers.Dense(dim * 3, name="qkv")
        self.proj = tf.keras.layers.Dense(dim, name="proj")

    def build(self, input_shape):
        self.built = True

    def call(self, x, mask=None):
        batch_windows = tf.shape(x)[0]
        tokens = tf.shape(x)[1]
        qkv = tf.reshape(self.qkv(x), [batch_windows, tokens, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        attention = tf.matmul(q * self.scale, k, transpose_b=True)
        if mask is not None:
            windows = tf.shape(mask)[0]
            attention = tf.reshape(attention, [-1, windows, self.num_heads, tokens, tokens])
            attention = attention + mask[None, :, None, :, :]
            attention = tf.reshape(attention, [-1, self.num_heads, tokens, tokens])
        attention = tf.nn.softmax(attention, axis=-1)
        x = tf.matmul(attention, v)
        x = tf.transpose(x, [0, 2, 1, 3])
        return self.proj(tf.reshape(x, [batch_windows, tokens, self.dim]))


class Mlp(tf.keras.layers.Layer):
    def __init__(self, dim, ratio, dropout, **kwargs):
        super().__init__(**kwargs)
        hidden = int(dim * ratio)
        self.fc1 = tf.keras.layers.Dense(hidden, name="fc1")
        self.fc2 = tf.keras.layers.Dense(dim, name="fc2")
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=None):
        x = self.dropout(tf.nn.gelu(self.fc1(x)), training=training)
        return self.dropout(self.fc2(x), training=training)


@tf.keras.utils.register_keras_serializable(package="NeuroLens")
class SwinTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0, mlp_ratio=4.0, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.dim, self.num_heads = dim, num_heads
        self.window_size, self.shift_size = window_size, shift_size
        self.mlp_ratio, self.dropout_rate = mlp_ratio, dropout
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm1")
        self.attn = WindowAttention(dim, num_heads, name="attn")
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="norm2")
        self.mlp = Mlp(dim, mlp_ratio, dropout, name="mlp")

    def build(self, input_shape):
        self.built = True

    def call(self, x, attention_mask=None, training=None):
        shortcut = x
        x = self.norm1(x)
        height, width = tf.shape(x)[1], tf.shape(x)[2]
        if self.shift_size:
            x = tf.roll(x, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        batch = tf.shape(x)[0]
        ws = self.window_size
        windows = tf.reshape(x, [batch, height // ws, ws, width // ws, ws, self.dim])
        windows = tf.transpose(windows, [0, 1, 3, 2, 4, 5])
        windows = tf.reshape(windows, [-1, ws * ws, self.dim])
        windows = self.attn(windows, mask=attention_mask)
        x = tf.reshape(windows, [batch, height // ws, width // ws, ws, ws, self.dim])
        x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
        x = tf.reshape(x, [batch, height, width, self.dim])
        if self.shift_size:
            x = tf.roll(x, shift=[self.shift_size, self.shift_size], axis=[1, 2])
        x = shortcut + x
        return x + self.mlp(self.norm2(x), training=training)

    def get_config(self):
        return {**super().get_config(), "dim": self.dim, "num_heads": self.num_heads,
                "window_size": self.window_size, "shift_size": self.shift_size,
                "mlp_ratio": self.mlp_ratio, "dropout": self.dropout_rate}


SWIN_CUSTOM_OBJECTS = {
    "PatchEmbedding": PatchEmbedding,
    "PatchMerging": PatchMerging,
    "SwinTransformerBlock": SwinTransformerBlock,
}
