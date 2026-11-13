import numpy as np
from typing import Tuple, Optional, Any, Dict
from app.core.logging import logger


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    """Matches notebook _normalize_cam: ReLU and scale to [0, 1]."""
    cam = np.maximum(cam, 0)
    mx = float(np.max(cam))
    return (cam / mx) if mx > 0 else cam


def build_cam_model(classifier: Any, conv_layer_name: str = "top_conv", tf_module: Any = None) -> Any:
    """Build sub-model outputting the feature map of the convolutional layer and model prediction."""
    tf = tf_module
    if tf is None:
        import tensorflow as tf
    try:
        conv_layer = classifier.get_layer(conv_layer_name)
    except Exception as e:
        # Find candidate convolutional layers
        conv_names = [l.name for l in classifier.layers if "conv" in l.name.lower()]
        raise ValueError(
            f"Layer '{conv_layer_name}' not found in classifier. "
            f"Available candidate conv layers: {conv_names[:8]}"
        ) from e

    return tf.keras.Model(classifier.inputs, [conv_layer.output, classifier.output])


def make_gradcam(
    classifier: Any,
    image_batch: np.ndarray,
    class_index: Optional[int] = None,
    conv_layer_name: str = "top_conv",
    tf_module: Any = None
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Notebook-exact Grad-CAM implementation.
    Returns (normalized_cam_2d, class_index, output_scores).
    Raises RuntimeError if gradients are unavailable.
    """
    tf = tf_module
    if tf is None:
        import tensorflow as tf

    cam_model = build_cam_model(classifier, conv_layer_name, tf_module=tf)
    x = tf.convert_to_tensor(image_batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv, scores = cam_model(x, training=False)
        if class_index is None:
            class_index = int(tf.argmax(scores[0]).numpy())
        target = scores[:, class_index]

    grads = tape.gradient(target, conv)
    if grads is None:
        raise RuntimeError(f"Gradients unavailable for Grad-CAM on layer '{conv_layer_name}'.")

    conv0 = conv[0]
    grads0 = grads[0]
    weights = tf.reduce_mean(grads0, axis=(0, 1))
    cam = tf.reduce_sum(conv0 * weights, axis=-1).numpy()

    return normalize_cam(cam), class_index, scores[0].numpy()


def make_gradcam_plus_plus(
    classifier: Any,
    image_batch: np.ndarray,
    class_index: Optional[int] = None,
    conv_layer_name: str = "top_conv",
    tf_module: Any = None
) -> Tuple[np.ndarray, int, np.ndarray]:
    """
    Notebook-exact Grad-CAM++ with higher-order gradients.
    Raises RuntimeError clearly if higher-order gradients are unsupported.
    """
    tf = tf_module
    if tf is None:
        import tensorflow as tf

    cam_model = build_cam_model(classifier, conv_layer_name, tf_module=tf)
    x = tf.convert_to_tensor(image_batch, dtype=tf.float32)

    with tf.GradientTape() as tape3:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                conv, scores = cam_model(x, training=False)
                if class_index is None:
                    class_index = int(tf.argmax(scores[0]).numpy())
                target = scores[:, class_index]
            first = tape1.gradient(target, conv)
        second = tape2.gradient(first, conv)
    third = tape3.gradient(second, conv)

    if first is None or second is None or third is None:
        raise RuntimeError(
            "Higher-order gradients unavailable in this TensorFlow/model setup for Grad-CAM++. "
            "Standard Grad-CAM can still be used."
        )

    conv0, g1, g2, g3 = conv[0], first[0], second[0], third[0]
    numerator = tf.square(g1)
    denominator = 2.0 * g2 + tf.reduce_sum(conv0 * g3, axis=(0, 1), keepdims=True)
    alpha = numerator / (denominator + tf.keras.backend.epsilon())
    weights = tf.reduce_sum(alpha * tf.nn.relu(g1), axis=(0, 1))
    cam = tf.reduce_sum(conv0 * weights, axis=-1).numpy()

    return normalize_cam(cam), class_index, scores[0].numpy()


def make_segmentation_gradcam(
    segmenter: Any,
    image_batch: np.ndarray,
    conv_layer_name: Optional[str] = None,
    tf_module: Any = None
) -> np.ndarray:
    """
    Compute segmentation XAI by targeting the foreground lesion output
    backpropagated to the bottleneck / latent conv layer.
    """
    tf = tf_module
    if tf is None:
        import tensorflow as tf

    # Determine target layer
    if not conv_layer_name:
        conv_names = [l.name for l in segmenter.layers if "conv" in l.name.lower()]
        if not conv_names:
            raise RuntimeError("No convolutional layer found in segmenter for segmentation XAI.")
        # Pick intermediate bottleneck convolution
        mid_idx = len(conv_names) // 2
        conv_layer_name = conv_names[mid_idx]

    try:
        conv_layer = segmenter.get_layer(conv_layer_name)
    except Exception as e:
        raise RuntimeError(f"Layer '{conv_layer_name}' not found in segmenter.") from e

    seg_cam_model = tf.keras.Model(segmenter.inputs, [conv_layer.output, segmenter.output])
    x = tf.convert_to_tensor(image_batch, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv, seg_out = seg_cam_model(x, training=False)
        # Target is foreground prediction sum
        target = tf.reduce_sum(seg_out)

    grads = tape.gradient(target, conv)
    if grads is None:
        raise RuntimeError(f"Gradients unavailable for Segmentation XAI on layer '{conv_layer_name}'.")

    conv0 = conv[0]
    grads0 = grads[0]
    weights = tf.reduce_mean(grads0, axis=(0, 1))
    cam = tf.reduce_sum(conv0 * weights, axis=-1).numpy()

    return normalize_cam(cam)


def compute_synthetic_heatmap_for_dev(width: int = 128, height: int = 128, seed: int = 42) -> np.ndarray:
    """Synthetic 2D heatmap for development / test doubles."""
    np.random.seed(seed)
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height * 0.45, width * 0.42
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    radius = min(width, height) * 0.32
    heatmap = np.clip(1.0 - dist / radius, 0, 1)
    return np.power(heatmap, 1.6).astype(np.float32)


def compute_synthetic_binary_mask_for_dev(width: int = 128, height: int = 128, seed: int = 42) -> np.ndarray:
    """Synthetic binary mask (1 for tumor lesion, 0 for background)."""
    heatmap = compute_synthetic_heatmap_for_dev(width, height, seed)
    return (heatmap >= 0.5).astype(bool)
