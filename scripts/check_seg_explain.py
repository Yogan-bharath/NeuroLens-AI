import os
import sys
import glob
import cv2
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.abspath("backend"))

from app.core.config import settings
from app.services.preprocessing_service import preprocessing_service


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


def feature_map_shape(model, layer_name: str):
    return tuple(int(d) for d in model.get_layer(layer_name).output.shape[1:])


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
    """degenerate=True -> all zeros/constant: report 'no explanation', never fall back to the mask."""
    cam = np.asarray(cam, np.float32)
    s = {"min": float(cam.min()), "max": float(cam.max()), "mean": float(cam.mean()),
         "hot_fraction": float((cam > 0.5).mean()),
         "degenerate": bool(cam.max() < 1e-6 or (cam.max() - cam.min()) < 1e-3)}
    if pred_mask is not None:
        m = np.asarray(pred_mask) > 0
        s["mask_area_fraction"] = float(m.mean())
        s["heat_mass_in_mask"] = float(cam[m].sum() / cam.sum()) if cam.sum() > 0 else 0.0
    return s


def composite_preview(mri_rgb_uint8: np.ndarray, rgba: np.ndarray, opacity: float = 0.65) -> np.ndarray:
    """For the debug script only: MRI + heatmap alpha-blended."""
    a = (rgba[..., 3:4].astype(np.float32) / 255.0) * opacity
    out = mri_rgb_uint8.astype(np.float32) * (1 - a) + rgba[..., :3].astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    model_path = os.path.join("backend", "model_artifacts", "deeplabv3plus.keras")
    if not os.path.exists(model_path):
        model_path = os.path.join("backend", "model_artifacts", "BRISC_Binary_UNet_best.keras")
    print(f"Loading model from: {model_path}")

    # Load without compiling
    model = tf.keras.models.load_model(model_path, compile=False)
    print(f"Model loaded successfully. Input: {model.input_shape}, Output: {model.output_shape}")

    # Verify candidate layers
    layer_names = [l.name for l in model.layers]
    default_layer = "conv2d_20" if "conv2d_20" in layer_names else None
    if not default_layer:
        # find layer with output 16x16x256
        for l in model.layers:
            if hasattr(l, "output_shape") and l.output_shape and len(l.output_shape) == 4:
                if l.output_shape[1:4] == (16, 16, 256):
                    default_layer = l.name
    print(f"Selected layer for diagnosis: {default_layer}")
    if default_layer:
        print(f"Layer feature map shape: {feature_map_shape(model, default_layer)}")

    # Setup debug storage
    debug_dir = os.path.join("backend", "storage", "debug")
    os.makedirs(debug_dir, exist_ok=True)

    # Find 3 to 5 real images
    candidates = glob.glob(os.path.join("backend", "storage", "uploads", "*.jpg"))
    if not candidates:
        candidates = glob.glob(os.path.join("backend", "storage", "uploads", "*.png"))
    sample_images = candidates[:5]
    print(f"Running check on {len(sample_images)} real images...")

    results = []
    for idx, img_path in enumerate(sample_images):
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        with open(img_path, "rb") as f:
            img_bytes = f.read()

        # Existing preprocessing for DeepLabV3+
        x_batch = preprocessing_service.preprocess_for_deeplabv3plus(img_bytes)
        pil_img = preprocessing_service.load_pil_rgb(img_bytes)
        mri_rgb = np.asarray(pil_img)
        h, w = mri_rgb.shape[:2]

        # Model prediction
        raw_pred = model(x_batch, training=False).numpy()[0]
        if raw_pred.ndim == 3 and raw_pred.shape[-1] == 1:
            prob_map = raw_pred.squeeze(-1)
        elif raw_pred.ndim == 3 and raw_pred.shape[-1] == 2:
            prob_map = raw_pred[:, :, 1]
        else:
            prob_map = raw_pred

        pred_mask_256 = (prob_map >= 0.5).astype(np.uint8)
        pred_mask = cv2.resize(pred_mask_256, (w, h), interpolation=cv2.INTER_NEAREST)

        # Seg-Grad-CAM on conv2d_20 mapped to (h, w)
        cam = seg_gradcam(model, x_batch, default_layer, out_size=(h, w))
        stats = heat_stats(cam, pred_mask)

        # Render outputs
        cam_gray_u8 = (cam * 255.0).astype(np.uint8)
        rgba = heatmap_rgba(cam)
        composite = composite_preview(mri_rgb, rgba, opacity=0.65)

        # Save debug artifacts
        cam_gray_path = os.path.join(debug_dir, f"sample_{idx}_{base_name}_raw_cam.png")
        rgba_path = os.path.join(debug_dir, f"sample_{idx}_{base_name}_heatmap_rgba.png")
        composite_path = os.path.join(debug_dir, f"sample_{idx}_{base_name}_composite.png")

        cv2.imwrite(cam_gray_path, cam_gray_u8)
        cv2.imwrite(rgba_path, cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        cv2.imwrite(composite_path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

        res_item = {
            "image": os.path.basename(img_path),
            "stats": stats,
            "raw_cam_path": cam_gray_path,
            "rgba_path": rgba_path,
            "composite_path": composite_path,
        }
        results.append(res_item)
        print(f"\n--- Image {idx + 1}: {os.path.basename(img_path)} ---")
        print(f"  Stats: {stats}")
        print(f"  Saved raw cam: {cam_gray_path}")
        print(f"  Saved heatmap RGBA: {rgba_path}")
        print(f"  Saved composite: {composite_path}")

    # Also check fine layer up_sampling2d_2 if present
    fine_layer = "up_sampling2d_2" if "up_sampling2d_2" in layer_names else None
    if fine_layer and len(sample_images) > 0:
        print(f"\n--- Fine Layer Check: {fine_layer} ---")
        img_path = sample_images[0]
        with open(img_path, "rb") as f:
            img_bytes = f.read()
        x_batch = preprocessing_service.preprocess_for_deeplabv3plus(img_bytes)
        cam_fine = seg_gradcam(model, x_batch, fine_layer, out_size=(256, 256))
        stats_fine = heat_stats(cam_fine)
        print(f"  Fine layer feature map: {feature_map_shape(model, fine_layer)}")
        print(f"  Fine layer stats: {stats_fine}")


if __name__ == "__main__":
    main()

