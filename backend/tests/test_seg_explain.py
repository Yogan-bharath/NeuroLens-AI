"""
Unit and integration tests for Seg-Grad-CAM 2D Tumor Segmentation explainability.
Uses small stand-in models so tests run in seconds without external weights or DB.
"""

import cv2
import numpy as np
import pytest
import tensorflow as tf

from app.ml.seg2d_explain import (
    seg_gradcam,
    feature_map_shape,
    heatmap_rgba,
    encode_png_rgba,
    heat_stats,
    generate_segmentation_explanation,
)


def create_standin_seg_model():
    """Lightweight stand-in model with a Conv2D layer followed by a sigmoid segmentation head."""
    inputs = tf.keras.Input(shape=(32, 32, 3), name="input_image")
    x = tf.keras.layers.Conv2D(8, (3, 3), padding="same", name="conv2d_20")(inputs)
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid", name="head")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def test_heatmap_rgba_properties():
    """Tests that heatmap_rgba properly maps values to Jet colormap and alpha transparency."""
    cam = np.zeros((10, 10), dtype=np.float32)
    cam[0, 0] = 0.0  # Cold
    cam[5, 5] = 1.0  # Hot
    cam[0, 9] = 0.1  # Low

    rgba = heatmap_rgba(cam)
    assert rgba.shape == (10, 10, 4)
    assert rgba.dtype == np.uint8

    # Cold pixel: alpha == 0
    assert rgba[0, 0, 3] == 0

    # Hot pixel: alpha == 255 and red dominant
    assert rgba[5, 5, 3] == 255
    r, g, b = rgba[5, 5, 0], rgba[5, 5, 1], rgba[5, 5, 2]
    assert r > b, f"Expected red > blue for peak saliency, got r={r}, b={b}"

    # Low value: blue dominant
    r_low, g_low, b_low = rgba[0, 9, 0], rgba[0, 9, 1], rgba[0, 9, 2]
    assert b_low > r_low, f"Expected blue > red for low saliency, got r={r_low}, b={b_low}"

    # PNG round-trip test
    png_bytes = encode_png_rgba(rgba)
    decoded = cv2.imdecode(np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded.shape == (10, 10, 4)
    assert decoded[0, 0, 3] == 0
    assert decoded[5, 5, 3] == 255


def test_heat_stats_degenerate_and_mask_fraction():
    """Tests heat_stats degenerate classification and heat_mass_in_mask calculation."""
    # 1. All-zero map -> degenerate True
    zero_cam = np.zeros((32, 32), dtype=np.float32)
    s_zero = heat_stats(zero_cam)
    assert s_zero["degenerate"] is True

    # 2. Constant map -> degenerate True
    const_cam = np.ones((32, 32), dtype=np.float32) * 0.5
    s_const = heat_stats(const_cam)
    assert s_const["degenerate"] is True

    # 3. Valid map with one hot square fully inside mask -> heat_mass_in_mask 1.0
    cam = np.zeros((32, 32), dtype=np.float32)
    mask = np.zeros((32, 32), dtype=np.uint8)

    # Hot square in region [10:20, 10:20]
    cam[10:20, 10:20] = 0.8
    # Mask covers [8:22, 8:22] (strictly enclosing the hot region)
    mask[8:22, 8:22] = 1

    s_mask = heat_stats(cam, pred_mask=mask)
    assert s_mask["degenerate"] is False
    assert pytest.approx(s_mask["heat_mass_in_mask"], 0.001) == 1.0
    assert s_mask["mask_area_fraction"] > 0


def test_seg_gradcam_on_standin_model():
    """Tests seg_gradcam execution on small stand-in model."""
    model = create_standin_seg_model()
    x_batch = np.random.uniform(0.0, 1.0, size=(1, 32, 32, 3)).astype(np.float32)

    cam = seg_gradcam(model, x_batch, layer_name="conv2d_20", out_size=(48, 48))

    assert cam.shape == (48, 48)
    assert cam.dtype == np.float32
    assert not np.isnan(cam).any()
    assert not np.isinf(cam).any()
    assert 0.0 <= cam.min()
    assert cam.max() <= 1.0


def test_generate_segmentation_explanation_contract():
    """Tests explanation response dictionary structure and contract."""
    model = create_standin_seg_model()
    x_batch = np.random.uniform(0.0, 1.0, size=(1, 32, 32, 3)).astype(np.float32)
    mask = np.zeros((32, 32), dtype=np.uint8)

    res = generate_segmentation_explanation(
        model=model,
        x_batch=x_batch,
        out_size=(32, 32),
        pred_mask=mask,
        preferred_layer="conv2d_20",
    )

    assert "available" in res
    assert "layer" in res
    assert res["layer"] == "conv2d_20"
    assert "map_shape" in res
    assert "method" in res
    assert res["method"] == "SEG-GRADCAM"
    assert "caption" in res
    assert "disclaimer" in res

    if res["available"]:
        assert res["heatmap_png"].startswith("data:image/png;base64,")
        assert res["stats"]["degenerate"] is False
    else:
        assert res["heatmap_png"] is None
        assert "reason" in res


@pytest.mark.asyncio
async def test_segmentation_service_explain_integration(monkeypatch):
    """Tests segmentation service with stubbed model produces valid explain fields."""
    from app.services.segmentation_2d_service import segmentation_2d_service

    standin_model = create_standin_seg_model()

    # Create dummy 32x32 image bytes (PNG)
    dummy_img = (np.random.uniform(0, 255, size=(32, 32, 3))).astype(np.uint8)
    _, buf = cv2.imencode(".png", dummy_img)
    img_bytes = buf.tobytes()

    monkeypatch.setattr(
        segmentation_2d_service,
        "_select_model",
        lambda: (standin_model, "deeplabv3plus_stub", lambda b: np.zeros((1, 32, 32, 3), dtype=np.float32)),
    )

    res = await segmentation_2d_service.segment_2d(
        image_bytes=img_bytes,
        analysis_id="test_analysis_123",
        explain=True,
    )

    assert "mask_url" in res
    assert "overlay_url" in res
    assert "explain" in res
    exp = res["explain"]
    assert exp["method"] == "SEG-GRADCAM"
    assert "layer" in exp
    assert "map_shape" in exp
    assert "stats" in exp
    assert "caption" in exp
    assert "disclaimer" in exp

    if exp["available"]:
        assert exp["heatmap_png"] is not None
        # Verify alpha channel is not constant in decoded heatmap
        b64_part = exp["heatmap_png"].split(",")[-1]
        import base64
        raw_bytes = base64.b64decode(b64_part)
        decoded = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        alpha = decoded[:, :, 3]
        assert alpha.shape == (32, 32)
        # Cold pixels have alpha 0, hot pixels have alpha > 0
        assert alpha.min() == 0
    else:
        assert exp["heatmap_png"] is None


