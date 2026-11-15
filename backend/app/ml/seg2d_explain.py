"""
Seg-Grad-CAM 2D Tumor Segmentation Explainability Module.
Computes genuine class-activation attribution for 2D DeepLabV3+ tumor segmentation.
"""

import base64
import cv2
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Tuple, Optional


def seg_gradcam(model, x_batch: np.ndarray, layer_name: str, out_size=(256, 256)) -> np.ndarray:
    """x_batch: (1,256,256,3) float32 in [0,1] (the segmentation endpoint's existing preprocessing).
    Target score = SUM of predicted tumor probability over all pixels. Returns float32 (H,W) in [0,1]."""
    layer = model.get_layer(layer_name)
    grad_model = tf.keras.Model(model.inputs, [layer.output, model.output])
    xt = tf.convert_to_tensor(x_batch, tf.float32)
    with tf.GradientTape() as tape:
        conv, pred_mask = grad_model(xt, training=False)
        target = tf.reduce_sum(pred_mask)
    grads = tape.gradient(target, conv)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))           # one weight per channel
    conv0 = conv[0]
    heat = tf.squeeze(conv0 @ pooled[..., tf.newaxis])       # weighted sum over channels
    heat = tf.maximum(heat, 0)                               # ReLU: keep positive contributions
    heat = (heat / (tf.reduce_max(heat) + 1e-8)).numpy()     # normalise to [0,1]
    out_h, out_w = out_size
    return cv2.resize(heat, (out_w, out_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def feature_map_shape(model, layer_name: str) -> Tuple[int, ...]:
    layer = model.get_layer(layer_name)
    return tuple(int(d) for d in layer.output.shape[1:])


def heatmap_rgba(cam: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """cam (H,W) in [0,1] -> uint8 RGBA. Colour = JET(cam); alpha = cam, so cold regions are transparent."""
    c = np.clip(cam, 0.0, 1.0) ** gamma
    c8 = (c * 255).astype(np.uint8)
    rgb = cv2.cvtColor(cv2.applyColorMap(c8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return np.dstack([rgb, c8])


def encode_png_rgba(rgba: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    if not ok:
        raise ValueError("PNG encoding failed")
    return buf.tobytes()


def heat_stats(cam: np.ndarray, pred_mask: np.ndarray = None) -> dict:
    """Computes attribution statistics, aligning the predicted mask to the CAM grid."""
    cam = np.asarray(cam, np.float32)
    s = {
        "min": float(cam.min()) if cam.size > 0 else 0.0,
        "max": float(cam.max()) if cam.size > 0 else 0.0,
        "mean": float(cam.mean()) if cam.size > 0 else 0.0,
        "hot_fraction": float((cam > 0.5).mean()) if cam.size > 0 else 0.0,
        "degenerate": bool(
            cam.size == 0
            or cam.max() < 1e-6
            or (cam.max() - cam.min()) < 1e-3
        ),
    }
    if pred_mask is not None:
        m = np.asarray(pred_mask).squeeze()
        if m.ndim != 2:
            raise ValueError(f"Expected a 2D predicted mask, got shape {m.shape}")
        m = m > 0
        if m.shape != cam.shape:
            if cam.size == 0:
                raise ValueError("Cannot align a predicted mask to an empty CAM")
            m = cv2.resize(
                m.astype(np.uint8),
                (cam.shape[1], cam.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        s["mask_area_fraction"] = float(m.mean())
        total_cam = float(cam.sum())
        s["heat_mass_in_mask"] = float(cam[m].sum() / total_cam) if total_cam > 0 else 0.0
    return s


def composite_preview(mri_rgb_uint8: np.ndarray, rgba: np.ndarray, opacity: float = 0.65) -> np.ndarray:
    """For preview/debug: MRI + heatmap alpha-blended."""
    a = (rgba[..., 3:4].astype(np.float32) / 255.0) * opacity
    out = mri_rgb_uint8.astype(np.float32) * (1 - a) + rgba[..., :3].astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def resolve_layer(model, preferred_layer: Optional[str] = None) -> Tuple[str, str]:
    """
    Resolves target layer name. Prefer the high-resolution decoder layer.
    """
    layer_names = [l.name for l in model.layers]
    if preferred_layer and preferred_layer in layer_names:
        return preferred_layer, "explicit"

    if "up_sampling2d_2" in layer_names:
        return "up_sampling2d_2", "default_fine_layer"

    if "conv2d_20" in layer_names:
        return "conv2d_20", "deep_layer_fallback"

    # Search for layer with shape (16, 16, 256)
    for l in reversed(model.layers):
        if hasattr(l, "output_shape") and l.output_shape and len(l.output_shape) == 4:
            if l.output_shape[1:4] == (16, 16, 256):
                return l.name, "shape_match_16x16x256"

    # Fallback to up_sampling2d_2 or first 4D layer
    if "up_sampling2d_2" in layer_names:
        return "up_sampling2d_2", "fine_layer_fallback"

    return layer_names[-2], "generic_fallback"


def generate_segmentation_explanation(
    model: Any,
    x_batch: np.ndarray,
    out_size: Tuple[int, int],
    pred_mask: Optional[np.ndarray] = None,
    preferred_layer: Optional[str] = "up_sampling2d_2",
) -> Dict[str, Any]:
    """
    Generates Seg-Grad-CAM explanation.
    Returns structured explanation dictionary complying with API contract.
    """
    target_layer, _ = resolve_layer(model, preferred_layer)
    shape_tuple = feature_map_shape(model, target_layer)
    map_shape_str = f"{shape_tuple[0]} × {shape_tuple[1]} × {shape_tuple[2]}" if len(shape_tuple) == 3 else str(shape_tuple)

    try:
        cam = seg_gradcam(model, x_batch, target_layer, out_size=out_size)
    except Exception as exc:
        return {
            "available": False,
            "reason": f"Seg-Grad-CAM computation failed on layer '{target_layer}': {exc}",
            "layer": target_layer,
            "map_shape": map_shape_str,
            "method": "SEG-GRADCAM",
            "stats": {"degenerate": True},
            "heatmap_png": None,
            "caption": "Shows features that influenced total predicted tumor probability. Grad-CAM may extend outside the predicted mask and is not a tumor boundary.",
            "disclaimer": "Research/educational use only. Not clinically validated.",
        }

    stats = heat_stats(cam, pred_mask)

    rgba = heatmap_rgba(cam)
    png_bytes = encode_png_rgba(rgba)
    b64_str = base64.b64encode(png_bytes).decode("ascii")
    heatmap_data_url = f"data:image/png;base64,{b64_str}"

    fine_note = " using the higher-resolution decoder layer" if target_layer == "up_sampling2d_2" else ""
    caption = (
        f"Shows features that influenced total predicted tumor probability{fine_note}. "
        "Grad-CAM may extend outside the predicted mask and is not a tumor boundary."
    )

    return {
        "available": True,
        "heatmap_png": heatmap_data_url,
        "heatmap_png_bytes": png_bytes,
        "layer": target_layer,
        "map_shape": map_shape_str,
        "stats": stats,
        "method": "SEG-GRADCAM",
        "caption": caption,
        "disclaimer": "Research/educational use only. Not clinically validated.",
    }

