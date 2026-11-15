import numpy as np
import pytest
import tensorflow as tf

from app.ml.attention_vit_2d_xai import make_2d_gradcam


def create_standin_model():
    inputs = tf.keras.Input(shape=(32, 32, 4))
    features = tf.keras.layers.Conv2D(8, 3, strides=2, padding="same")(inputs)
    features = tf.keras.layers.Activation("relu", name="bottleneck_relu2")(features)
    outputs = tf.keras.layers.Conv2D(4, 1, activation="softmax")(features)
    return tf.keras.Model(inputs, outputs)


def test_2d_attention_vit_gradcam_returns_class_specific_input_size_maps():
    model = create_standin_model()
    image_slice = np.random.default_rng(42).normal(size=(32, 32, 4)).astype(np.float32)

    maps = [
        make_2d_gradcam(model, image_slice, class_index)
        for class_index in (None, 1, 2, 3)
    ]

    for heatmap in maps:
        assert heatmap.shape == image_slice.shape[:2]
        assert heatmap.dtype == np.float32
        assert np.isfinite(heatmap).all()
        assert 0.0 <= heatmap.min() <= heatmap.max() <= 1.0


def test_2d_attention_vit_gradcam_rejects_out_of_range_class():
    model = create_standin_model()
    image_slice = np.zeros((32, 32, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="outside the model output channels"):
        make_2d_gradcam(model, image_slice, class_index=4)
