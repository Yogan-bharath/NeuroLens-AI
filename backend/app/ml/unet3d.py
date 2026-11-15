"""
3D U-Net (best_3d_unet_model.keras): preprocessing, sliding-window inference, 3D Grad-CAM, uncertainty.
Extracted from the braingliomav6 notebook. The Grad-CAM resize is FIXED (see resize_trilinear).

Model: input (None, 96, 96, 64, 4) float32, axes (H, W, D, C) in the native NIfTI array order
(nib.load(path).get_fdata(), NO reorientation; D = axial slice axis). Channels: FLAIR, T1, T1CE, T2.
Output (None, 96, 96, 64, 4) softmax. Classes: 0 background, 1 necrotic/non-enhancing, 2 edema,
3 enhancing (BraTS label 4 was remapped to 3 during training).
Load with: keras.models.load_model(path, compile=False, custom_objects=CUSTOM_OBJECTS)
"""
import numpy as np
import tensorflow as tf

PATCH_SHAPE = (96, 96, 64)
STRIDE = (64, 64, 32)
MODALITIES = ("flair", "t1", "t1ce", "t2")
NUM_CLASSES = 4
CLASS_NAMES = {0: "Background", 1: "Necrotic / Non-enhancing", 2: "Edema", 3: "Enhancing Tumor"}
GRADCAM_LAYER = "bottleneck_relu2"  # (12, 12, 8, 256) for a 96x96x64 patch


# ---- stubs so load_model(...) never fails on the notebook's custom metric/loss names ----
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
        gt, pr = tf.equal(y_true, cls), tf.equal(y_pred, cls)
        inter = tf.reduce_sum(tf.cast(gt & pr, tf.float32), axis=[1, 2, 3])
        union = tf.reduce_sum(tf.cast(gt | pr, tf.float32), axis=[1, 2, 3])
        ious.append((inter + 1e-6) / (union + 1e-6))
    return tf.reduce_mean(tf.stack(ious, axis=0))


CUSTOM_OBJECTS = {"foreground_dice": foreground_dice, "loss_fn": loss_fn,
                  "mean_iou_foreground": mean_iou_foreground}


# ---- loading + normalisation (verbatim logic from the notebook) ----
def load_nifti(path):
    import nibabel as nib  # imported lazily so the rest of this module has no nibabel dependency
    return nib.load(path).get_fdata().astype(np.float32)  # NO reorientation, as in training


def normalize_modality(volume: np.ndarray) -> np.ndarray:
    """z-score over NON-ZERO voxels only; zeros stay zero."""
    mask = volume != 0
    if not np.any(mask):
        return volume.astype(np.float32)
    values = volume[mask]
    mean, std = values.mean(), values.std()
    if std < 1e-6:
        std = 1.0
    out = np.zeros_like(volume, dtype=np.float32)
    out[mask] = (volume[mask] - mean) / std
    return out


def prepare_volume(flair, t1, t1ce, t2) -> np.ndarray:
    """Four (H, W, D) arrays -> (H, W, D, 4) float32 in the order the model was trained on."""
    vols = [np.asarray(v, dtype=np.float32) for v in (flair, t1, t1ce, t2)]
    if any(v.ndim != 3 for v in vols) or len({v.shape for v in vols}) != 1:
        raise ValueError(f"All four modalities must be 3D with identical shapes, got {[v.shape for v in vols]}")
    return np.stack([normalize_modality(v) for v in vols], axis=-1).astype(np.float32)


# ---- sliding-window inference ----
def make_starts(length: int, patch: int, stride: int):
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts


def pad_to_min(image: np.ndarray, patch_shape=PATCH_SHAPE):
    """Zero-pad spatial dims (at the end) so every dim >= patch. Returns (padded, original_spatial_shape)."""
    orig = image.shape[:3]
    pads = [(0, max(p - s, 0)) for s, p in zip(orig, patch_shape)] + [(0, 0)]
    if any(a or b for a, b in pads):
        image = np.pad(image, pads, mode="constant")
    return image, orig


def sliding_window_predict(model, image, patch_shape=PATCH_SHAPE, stride=STRIDE,
                           batch_size=2, progress_cb=None, num_classes=NUM_CLASSES):
    """
    image (H, W, D, 4) -> softmax volume (H, W, D, num_classes), overlapping patches averaged.
    Same windows and averaging as the notebook; adds zero-padding for small volumes and batching.
    progress_cb(done, total) is called after each batch (use it for job progress).
    """
    image, orig = pad_to_min(image, patch_shape)
    H, W, D, _ = image.shape
    ph, pw, pd = patch_shape
    starts = [(z, y, x)
              for z in make_starts(H, ph, stride[0])
              for y in make_starts(W, pw, stride[1])
              for x in make_starts(D, pd, stride[2])]
    prob_sum = np.zeros((H, W, D, num_classes), np.float32)
    count = np.zeros((H, W, D, 1), np.float32)
    for i in range(0, len(starts), batch_size):
        chunk = starts[i:i + batch_size]
        batch = np.stack([image[z:z + ph, y:y + pw, x:x + pd] for z, y, x in chunk]).astype(np.float32)
        pred = model(batch, training=False).numpy()
        for (z, y, x), p in zip(chunk, pred):
            prob_sum[z:z + ph, y:y + pw, x:x + pd] += p
            count[z:z + ph, y:y + pw, x:x + pd] += 1.0
        if progress_cb:
            progress_cb(min(i + batch_size, len(starts)), len(starts))
    probs = prob_sum / np.maximum(count, 1.0)
    oh, ow, od = orig
    return probs[:oh, :ow, :od]


def labels_from_probs(probs: np.ndarray) -> np.ndarray:
    return np.argmax(probs, axis=-1).astype(np.uint8)


# ---- uncertainty ----
def prediction_entropy(probs: np.ndarray) -> np.ndarray:
    """Normalised voxel-wise entropy in [0, 1]."""
    p = np.clip(np.asarray(probs, np.float32), 1e-8, 1.0)
    return (-np.sum(p * np.log(p), axis=-1) / np.log(p.shape[-1])).astype(np.float32)


# ---- 3D Grad-CAM (fixed) ----
def _interp_matrix(n_in: int, n_out: int) -> np.ndarray:
    # half-pixel centres: identical to tf.image.resize(method="bilinear", antialias=False) when upsampling
    src = np.clip((np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5, 0, n_in - 1)
    i0 = np.floor(src).astype(int)
    i1 = np.minimum(i0 + 1, n_in - 1)
    w1 = (src - i0).astype(np.float32)
    m = np.zeros((n_out, n_in), np.float32)
    m[np.arange(n_out), i0] += 1.0 - w1
    m[np.arange(n_out), i1] += w1
    return m


def resize_trilinear(vol: np.ndarray, out_shape) -> np.ndarray:
    """
    Separable linear resize of an N-D array to out_shape, one axis at a time.
    This REPLACES the notebook's make_3d_gradcam resize, which used tf.image.resize on a 3D array
    (first axis treated as batch) and returned (96, 96, 96) instead of (96, 96, 64), mis-scaling depth.
    """
    out = np.asarray(vol, np.float32)
    for axis, n_out in enumerate(out_shape):
        m = _interp_matrix(out.shape[axis], n_out)
        out = np.moveaxis(np.tensordot(m, out, axes=([1], [axis])), 0, axis)
    return out


def gradcam3d(model, patch: np.ndarray, class_indices=(1, 2, 3), layer_name: str = GRADCAM_LAYER) -> dict:
    """
    Class-specific 3D Grad-CAM for one (H, W, D, 4) patch. Returns {class_index: float32 (H, W, D) in [0, 1]}.
    score_c = mean over the patch of softmax[..., c]; weights = mean gradient over the 3 spatial axes.
    One forward pass, one gradient per class.
    """
    grad_model = tf.keras.Model(model.inputs, [model.get_layer(layer_name).output, model.output])
    x = tf.convert_to_tensor(patch[None].astype(np.float32))
    maps = {}
    with tf.GradientTape(persistent=True) as tape:
        acts, preds = grad_model(x, training=False)
        scores = {c: tf.reduce_mean(preds[..., c]) for c in class_indices}
    try:
        for c in class_indices:
            grads = tape.gradient(scores[c], acts)
            if grads is None:
                raise RuntimeError("Gradients could not be computed")
            weights = tf.reduce_mean(grads, axis=(1, 2, 3))[0]
            cam = tf.nn.relu(tf.reduce_sum(acts[0] * weights[None, None, None, :], axis=-1)).numpy()
            cam = resize_trilinear(cam, patch.shape[:3])
            assert cam.shape == tuple(patch.shape[:3]), (cam.shape, patch.shape)
            m = cam.max()
            maps[c] = (cam / m if m > 0 else cam).astype(np.float32)
    finally:
        del tape
    return maps


# ---- choosing and placing the Grad-CAM patch WITHOUT ground truth ----
def tumor_center_start(pred_labels: np.ndarray, patch_shape=PATCH_SHAPE):
    """
    Patch start (z0, y0, x0) centred on the median of PREDICTED tumor voxels, clipped inside the volume.
    Returns None when nothing is predicted as tumor (-> report Grad-CAM unavailable).
    The volume must already be padded to >= patch size (pad_to_min).
    """
    coords = np.argwhere(pred_labels > 0)
    if len(coords) == 0:
        return None
    center = np.median(coords, axis=0).astype(int)
    return tuple(int(np.clip(c - p // 2, 0, max(s - p, 0)))
                 for c, p, s in zip(center, patch_shape, pred_labels.shape))


def extract_patch(image: np.ndarray, start, patch_shape=PATCH_SHAPE) -> np.ndarray:
    z, y, x = start
    ph, pw, pd = patch_shape
    return image[z:z + ph, y:y + pw, x:x + pd]


def embed_patch_map(patch_map: np.ndarray, start, volume_shape) -> np.ndarray:
    """Place a patch-level heatmap back into the full-volume grid (zeros outside the patch)."""
    full = np.zeros(volume_shape, np.float32)
    z, y, x = start
    ph, pw, pd = patch_map.shape
    full[z:z + ph, y:y + pw, x:x + pd] = patch_map
    return full

