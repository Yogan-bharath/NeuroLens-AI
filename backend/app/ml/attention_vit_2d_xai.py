"""Class-specific 2D Grad-CAM for the multimodal Attention CNN + ViT model."""

from typing import Optional

import numpy as np
import tensorflow as tf


def make_2d_gradcam(
    model: tf.keras.Model,
    image_slice: np.ndarray,
    class_index: Optional[int],
    layer_name: str = "bottleneck_relu2",
) -> np.ndarray:
    """Return one normalized Grad-CAM for an H×W×4 input slice.

    ``class_index=None`` targets the combined foreground probability; otherwise
    the selected tumor-class probability is used, matching the notebook's method.
    """
    image_slice = np.asarray(image_slice, dtype=np.float32)
    if image_slice.ndim != 3:
        raise ValueError(f"Expected an H×W×C slice, got shape {image_slice.shape}")
    if image_slice.shape[-1] != model.input_shape[-1]:
        raise ValueError(
            f"Slice has {image_slice.shape[-1]} channels, but the model expects "
            f"{model.input_shape[-1]}"
        )

    layer = model.get_layer(layer_name)
    grad_model = tf.keras.Model(model.inputs, [layer.output, model.output])
    input_tensor = tf.convert_to_tensor(image_slice[None, ...], dtype=tf.float32)

    with tf.GradientTape() as tape:
        activations, predictions = grad_model(input_tensor, training=False)
        if class_index is None:
            if predictions.shape[-1] is not None and predictions.shape[-1] < 2:
                raise ValueError("Foreground Grad-CAM requires background and tumor classes")
            target_score = tf.reduce_mean(tf.reduce_sum(predictions[..., 1:], axis=-1))
        else:
            if class_index < 0 or (
                predictions.shape[-1] is not None and class_index >= predictions.shape[-1]
            ):
                raise ValueError(f"Class index {class_index} is outside the model output channels")
            target_score = tf.reduce_mean(predictions[..., class_index])

    gradients = tape.gradient(target_score, activations)
    if gradients is None:
        raise RuntimeError(f"No gradients connect '{layer_name}' to the selected prediction")
    if activations.shape.rank != 4:
        raise ValueError(
            f"Expected a spatial 2D feature map from '{layer_name}', got shape {activations.shape}"
        )

    channel_weights = tf.reduce_mean(gradients, axis=(1, 2))
    heatmap = tf.reduce_sum(
        activations[0] * channel_weights[0][tf.newaxis, tf.newaxis, :],
        axis=-1,
    )
    heatmap = tf.maximum(heatmap, 0.0)
    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        image_slice.shape[:2],
        method="bilinear",
    )[..., 0]
    heatmap = tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap))
    return heatmap.numpy().astype(np.float32)
