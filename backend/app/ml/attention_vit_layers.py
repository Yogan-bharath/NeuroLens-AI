import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="NeuroLens", name="AttentionGate")
@tf.keras.utils.register_keras_serializable(name="AttentionGate")
class AttentionGate(tf.keras.layers.Layer):
    """Attention gate serialized with the supplied Attention CNN + ViT model."""
    def __init__(self, filters: int, **kwargs):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.theta = tf.keras.layers.Conv2D(self.filters, 1, use_bias=False, name="theta")
        self.phi = tf.keras.layers.Conv2D(self.filters, 1, use_bias=False, name="phi")
        self.psi = tf.keras.layers.Conv2D(1, 1, activation="sigmoid", name="psi")

    def build(self, input_shape):
        self.built = True

    def call(self, x, gating):
        attention = self.psi(tf.nn.relu(self.theta(x) + self.phi(gating)))
        return x * attention

    def get_config(self):
        return {**super().get_config(), "filters": self.filters}


ATTENTION_VIT_CUSTOM_OBJECTS = {
    "AttentionGate": AttentionGate,
    "NeuroLens>AttentionGate": AttentionGate,
}
