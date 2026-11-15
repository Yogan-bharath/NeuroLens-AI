"""
BraTS 3D U-Net Model Architecture and Research Utilities
Faithfully reproduced from research notebook: braingliomav6 (2).ipynb

Core Configurations:
- Framework: TensorFlow / Keras
- Input shape: (96, 96, 64, 4)
- Patch size: (96, 96, 64)
- Channels: 4 (FLAIR, T1, T1CE, T2)
- Classes: 4 (0: Background, 1: Necrotic, 2: Edema, 3: Enhancing)
- Base filters: 32
- Target Grad-CAM layer: bottleneck_relu2
- Stride for inference: (64, 64, 32)
"""

import os
from typing import Any, Dict, List, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

IMG_H, IMG_W, IMG_D = 96, 96, 64
PATCH_SIZE = (IMG_H, IMG_W, IMG_D)
N_CHANNELS = 4
NUM_CLASSES = 4
BASE_FILTERS = 32
GRAD_CAM_LAYER = "bottleneck_relu2"
INFER_STRIDE = (64, 64, 32)
MODALITY_NAMES = ["FLAIR", "T1", "T1CE", "T2"]

CLASS_NAMES = {
    0: "Background",
    1: "Necrotic / Non-enhancing",
    2: "Edema",
    3: "Enhancing Tumor"
}


# =========================================================================
# 1. EXACT MODEL ARCHITECTURE (From Cell 22)
# =========================================================================

def conv3d_block(x, filters: int, name: str):
    """
    Two consecutive 3x3x3 Conv3D layers with BatchNormalization and ReLU.
    padding='same', use_bias=False.
    Layer names match the notebook exactly, ending in {name}_relu2.
    """
    x = layers.Conv3D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name}_bn1")(x)
    x = layers.Activation("relu", name=f"{name}_relu1")(x)

    x = layers.Conv3D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name}_bn2")(x)
    x = layers.Activation("relu", name=f"{name}_relu2")(x)
    return x


def encoder_block(x, filters: int, name: str):
    features = conv3d_block(x, filters, name)
    pooled = layers.MaxPooling3D(2, name=f"{name}_pool")(features)
    return features, pooled


def decoder_block(x, skip, filters: int, name: str):
    x = layers.Conv3DTranspose(filters, 2, strides=2, padding="same", name=f"{name}_up")(x)
    x = layers.Concatenate(name=f"{name}_concat")([x, skip])
    return conv3d_block(x, filters, name)


def build_3d_unet(input_shape=(*PATCH_SIZE, N_CHANNELS), num_classes: int = NUM_CLASSES, base_filters: int = BASE_FILTERS) -> Model:
    """
    Builds the 3D U-Net model matching braingliomav6 (2).ipynb.
    Total parameters: ~5,606,628
    """
    inp = layers.Input(input_shape, name="mri_3d_input")

    s1, p1 = encoder_block(inp, base_filters, "enc1")          # 32 filters
    s2, p2 = encoder_block(p1, base_filters * 2, "enc2")      # 64 filters
    s3, p3 = encoder_block(p2, base_filters * 4, "enc3")      # 128 filters

    b = conv3d_block(p3, base_filters * 8, "bottleneck")      # 256 filters, has bottleneck_relu2

    d3 = decoder_block(b, s3, base_filters * 4, "dec3")       # 128 filters
    d2 = decoder_block(d3, s2, base_filters * 2, "dec2")       # 64 filters
    d1 = decoder_block(d2, s1, base_filters, "dec1")           # 32 filters

    out = layers.Conv3D(num_classes, 1, activation="softmax", name="segmentation_output")(d1)
    return Model(inp, out, name="3D_UNet_BraTS")


# =========================================================================
# 2. CUSTOM METRICS & LOSS FUNCTION (From Cell 24)
# =========================================================================

def foreground_dice(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.int32)
    one_hot = tf.one_hot(y_true, NUM_CLASSES)

    inter = tf.reduce_sum(one_hot * y_pred, axis=[1, 2, 3])
    denom = tf.reduce_sum(one_hot + y_pred, axis=[1, 2, 3])

    dice = (2.0 * inter + smooth) / (denom + smooth)
    return tf.reduce_mean(dice[:, 1:])


def loss_fn(y_true, y_pred):
    ce = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred))
    return ce + (1.0 - foreground_dice(y_true, y_pred))


def mean_iou_foreground(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)

    ious = []
    for cls in range(1, NUM_CLASSES):
        gt = tf.equal(y_true, cls)
        pr = tf.equal(y_pred, cls)
        inter = tf.reduce_sum(tf.cast(gt & pr, tf.float32), axis=[1, 2, 3])
        union = tf.reduce_sum(tf.cast(gt | pr, tf.float32), axis=[1, 2, 3])
        ious.append((inter + 1e-6) / (union + 1e-6))

    return tf.reduce_mean(tf.stack(ious, axis=0))


BRATS_3D_CUSTOM_OBJECTS = {
    "foreground_dice": foreground_dice,
    "loss_fn": loss_fn,
    "mean_iou_foreground": mean_iou_foreground,
}


# =========================================================================
# 3. SLIDING-WINDOW INFERENCE (From Cell 34)
# =========================================================================

def make_starts(length: int, patch: int, stride: int) -> List[int]:
    """Generates 1D patch starting coordinates with boundary alignment."""
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def sliding_window_predict(model: Any, image: np.ndarray, stride: Tuple[int, int, int] = INFER_STRIDE) -> np.ndarray:
    """
    Sliding window inference matching Cell 34.
    Includes production padding safeguard for volumes smaller than patch size.
    """
    H, W, D, C = image.shape
    ph, pw, pd = PATCH_SIZE

    # Safe padding if any dimension is smaller than patch size
    pad_h = max(0, ph - H)
    pad_w = max(0, pw - W)
    pad_d = max(0, pd - D)

    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, pad_d), (0, 0)), mode="constant", constant_values=0)
        padded_H, padded_W, padded_D, _ = image.shape
    else:
        padded_H, padded_W, padded_D = H, W, D

    z_starts = make_starts(padded_H, ph, stride[0])
    y_starts = make_starts(padded_W, pw, stride[1])
    x_starts = make_starts(padded_D, pd, stride[2])

    prob_sum = np.zeros((padded_H, padded_W, padded_D, NUM_CLASSES), dtype=np.float32)
    count = np.zeros((padded_H, padded_W, padded_D, 1), dtype=np.float32)

    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                patch = image[z0:z0 + ph, y0:y0 + pw, x0:x0 + pd, :]
                pred = model(patch[None].astype(np.float32), training=False)
                if hasattr(pred, "numpy"):
                    pred = pred.numpy()
                pred = pred[0]

                prob_sum[z0:z0 + ph, y0:y0 + pw, x0:x0 + pd, :] += pred
                count[z0:z0 + ph, y0:y0 + pw, x0:x0 + pd, :] += 1.0

    result = prob_sum / np.maximum(count, 1.0)

    # Remove padding if applied
    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        result = result[:H, :W, :D, :]

    return result


# =========================================================================
# 4. EVALUATION METRICS (From Cell 36)
# =========================================================================

def dice_iou_binary(gt: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    inter = float(np.logical_and(gt, pred).sum())
    union = float(np.logical_or(gt, pred).sum())

    dice = (2.0 * inter + 1e-6) / (float(gt.sum()) + float(pred.sum()) + 1e-6)
    iou = (inter + 1e-6) / (union + 1e-6)
    return float(dice), float(iou)


def evaluate_volume(gt: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    """
    Computes exact evaluation metrics from notebook Cell 36.
    Uses notebook region definitions:
    - Whole Tumor: labels > 0
    - Tumor Core: labels 1 and 3
    - Necrotic: label 1
    - Edema: label 2
    - Enhancing: label 3
    """
    result = {}

    for cls, name in [(1, "necrotic"), (2, "edema"), (3, "enhancing")]:
        d, i = dice_iou_binary(gt == cls, pred == cls)
        result[f"{name}_dice"] = round(d, 4)
        result[f"{name}_iou"] = round(i, 4)

    # Whole Tumor (labels > 0)
    d, i = dice_iou_binary(gt > 0, pred > 0)
    result["whole_tumor_dice"] = round(d, 4)
    result["whole_tumor_iou"] = round(i, 4)

    # Tumor Core (labels 1 and 3)
    d, i = dice_iou_binary(np.isin(gt, [1, 3]), np.isin(pred, [1, 3]))
    result["tumor_core_dice"] = round(d, 4)
    result["tumor_core_iou"] = round(i, 4)

    result["mean_foreground_dice"] = round(float(np.mean([
        result["necrotic_dice"], result["edema_dice"], result["enhancing_dice"]
    ])), 4)
    result["mean_foreground_iou"] = round(float(np.mean([
        result["necrotic_iou"], result["edema_iou"], result["enhancing_iou"]
    ])), 4)
    result["pixel_accuracy"] = round(float(np.mean(gt == pred)), 4)

    return result


# =========================================================================
# 5. EXPLAINABILITY SUITE (From Cells 44, 45, 48, 51, 56)
# =========================================================================

def extract_tumor_center_patch(image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int]]:
    """
    Extracts a deterministic tumor-centered patch matching Cell 45.
    Returns (patch_image, patch_mask, (z0, y0, x0)).
    """
    ph, pw, pd = PATCH_SIZE
    H, W, D = mask.shape

    max_z = max(H - ph, 0)
    max_y = max(W - pw, 0)
    max_x = max(D - pd, 0)

    tumor_coords = np.argwhere(mask > 0)
    if len(tumor_coords) == 0:
        center = np.array([H // 2, W // 2, D // 2])
    else:
        center = np.median(tumor_coords, axis=0).astype(int)

    z0 = int(np.clip(center[0] - ph // 2, 0, max_z))
    y0 = int(np.clip(center[1] - pw // 2, 0, max_y))
    x0 = int(np.clip(center[2] - pd // 2, 0, max_x))

    patch_image = image[z0:z0 + ph, y0:y0 + pw, x0:x0 + pd, :]
    patch_mask = mask[z0:z0 + ph, y0:y0 + pw, x0:x0 + pd]

    return patch_image, patch_mask, (z0, y0, x0)


def _interp_matrix(n_in: int, n_out: int) -> np.ndarray:
    src = np.clip((np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5, 0, n_in - 1)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, n_in - 1)
    w1 = (src - i0).astype(np.float32)
    m = np.zeros((n_out, n_in), np.float32)
    m[np.arange(n_out), i0] += 1.0 - w1
    m[np.arange(n_out), i1] += w1
    return m


def resize_trilinear(vol: np.ndarray, out_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Separable trilinear resize of a 3D volume to out_shape (H, W, D).
    Fixes the notebook bug where depth was mis-scaled by tf.image.resize.
    """
    out = np.asarray(vol, np.float32)
    for axis, n_out in enumerate(out_shape):
        m = _interp_matrix(out.shape[axis], n_out)
        out = np.moveaxis(np.tensordot(m, out, axes=([1], [axis])), 0, axis)
    return out


def make_3d_gradcam(model: Any, patch: np.ndarray, class_index: int, layer_name: str = GRAD_CAM_LAYER) -> np.ndarray:
    """
    Computes 3D Grad-CAM targeting bottleneck_relu2 matching Cell 44 with fixed trilinear resize.
    Returns normalized 3D heatmap of shape (H, W, D) with values in [0, 1].
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(layer_name).output,
            model.output
        ]
    )

    input_tensor = tf.convert_to_tensor(patch[None].astype(np.float32))

    with tf.GradientTape() as tape:
        activations, predictions = grad_model(input_tensor)
        class_score = tf.reduce_mean(predictions[..., class_index])

    gradients = tape.gradient(class_score, activations)
    if gradients is None:
        return np.zeros(patch.shape[:3], dtype=np.float32)

    # Average gradients over spatial dimensions (height, width, depth)
    weights = tf.reduce_mean(gradients, axis=(1, 2, 3))[0]
    activations = activations[0]

    # Weighted sum of feature channels
    heatmap = tf.reduce_sum(activations * weights[None, None, None, :], axis=-1)
    heatmap = tf.maximum(heatmap, 0).numpy()

    # Trilinear resize of bottleneck feature map back to patch spatial shape
    heatmap_full = resize_trilinear(heatmap, patch.shape[:3])

    max_val = np.max(heatmap_full)
    if max_val > 0:
        heatmap_full = heatmap_full / max_val

    return heatmap_full.astype(np.float32)


def extract_bottleneck_features(model: Any, patch: np.ndarray, layer_name: str = GRAD_CAM_LAYER) -> Dict[str, Any]:
    """
    Extracts bottleneck features matching Cell 48 & 49.
    Returns mean feature activation slice and representative channels.
    """
    feature_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=model.get_layer(layer_name).output
    )

    bottleneck_features = feature_model(patch[None].astype(np.float32), training=False).numpy()[0]
    num_channels = bottleneck_features.shape[-1]
    feature_z = bottleneck_features.shape[2] // 2

    # Mean feature map across all channels
    mean_feature = np.mean(bottleneck_features, axis=-1)
    min_val, max_val = float(mean_feature.min()), float(mean_feature.max())
    norm_mean = (mean_feature - min_val) / (max_val - min_val + 1e-8)

    # 4 representative channels
    channel_indices = np.linspace(0, num_channels - 1, 4, dtype=int).tolist()
    channels_data = {}
    for ch in channel_indices:
        ch_slice = bottleneck_features[:, :, feature_z, ch]
        c_min, c_max = float(ch_slice.min()), float(ch_slice.max())
        norm_ch = (ch_slice - c_min) / (c_max - c_min + 1e-8)
        channels_data[f"channel_{ch}"] = norm_ch.tolist()

    return {
        "feature_shape": list(bottleneck_features.shape),
        "total_channels": int(num_channels),
        "feature_z_slice": int(feature_z),
        "mean_feature_slice": norm_mean[:, :, feature_z].tolist(),
        "channels": channels_data,
    }


def ablate_modality(image: np.ndarray, modality_index: int) -> np.ndarray:
    """Zeroes out one modality channel matching Cell 51."""
    ablated = image.copy().astype(np.float32)
    ablated[..., modality_index] = 0.0
    return ablated


def prediction_entropy(probabilities: np.ndarray) -> np.ndarray:
    """
    Normalized predictive entropy matching Cell 56.
    Returns array of same spatial dimensions with entropy in [0, 1].
    """
    probs = np.clip(probabilities.astype(np.float32), 1e-8, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=-1)
    max_entropy = np.log(probs.shape[-1])
    return (entropy / max_entropy).astype(np.float32)

