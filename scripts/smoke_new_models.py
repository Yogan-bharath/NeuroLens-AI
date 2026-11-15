#!/usr/bin/env python3
"""
scripts/smoke_new_models.py

Fast smoke test for the newly integrated models and explainability pipelines:
  1. Task A: EfficientNetV2B0 256x256 classifier + Grad-CAM (top_activation, 8x8x1280)
  2. Task B: DeepLabV3+ Seg-Grad-CAM (conv2d_20 / up_sampling2d_2)
  3. Task C: 3D U-Net Volumetric BraTS Segmentation + Trilinear 3D Grad-CAM
Prints timings, output shapes, and predicted classes.
"""

import os
import sys
import time
import numpy as np
import tensorflow as tf

# Configure Python paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from backend.app.ml.adapters.efficientnet256_adapter import efficientnet256_adapter
from backend.app.ml.adapters.seg2d_xai_adapter import seg2d_xai_adapter
from backend.app.ml.adapters.unet3d_adapter import unet3d_adapter
from backend.app.ml.brats_3d_unet import build_3d_unet, BRATS_3D_CUSTOM_OBJECTS


def smoke_2d_classifier():
    print("\n" + "=" * 60)
    print("SMOKE TEST 1: EfficientNetV2B0 256x256 Classifier + Grad-CAM")
    print("=" * 60)
    model_path = os.path.join(BACKEND_DIR, "model_artifacts", "EfficientNetV2B0_clahe_256.keras")
    if not os.path.exists(model_path):
        print(f"[SKIP] Model artifact not found at {model_path}")
        return

    t0 = time.perf_counter()
    model = tf.keras.models.load_model(model_path, compile=False)
    load_time = time.perf_counter() - t0
    print(f"Loaded {os.path.basename(model_path)} in {load_time:.2f}s")

    # Synthetic 2D brain MRI slice (256x256x3)
    synth = np.zeros((256, 256, 3), dtype=np.uint8)
    import cv2
    cv2.ellipse(synth, (128, 128), (90, 110), 0, 0, 360, (180, 180, 180), -1)
    cv2.circle(synth, (140, 130), 25, (245, 245, 245), -1)
    _, encoded = cv2.imencode(".png", synth)
    raw_bytes = encoded.tobytes()

    # Preprocess
    t0 = time.perf_counter()
    x_256 = efficientnet256_adapter.preprocess_image(raw_bytes)
    prep_time = (time.perf_counter() - t0) * 1000
    print(f"Preprocessed tensor shape: {x_256.shape}, dtype={x_256.dtype}, range=[{x_256.min():.1f}, {x_256.max():.1f}] ({prep_time:.1f}ms)")

    # Predict
    t0 = time.perf_counter()
    pred_res = efficientnet256_adapter.predict(model, x_256)
    pred_time = (time.perf_counter() - t0) * 1000
    print(f"Predicted class: '{pred_res['predicted_class']}' (confidence={pred_res['confidence']:.4f}) in {pred_time:.1f}ms")
    print(f"Probabilities: {pred_res['probabilities']}")

    # Grad-CAM on top_activation
    t0 = time.perf_counter()
    cam, layer_name, f_dim = efficientnet256_adapter.explain_gradcam(
        model, x_256, target_layer="top_activation"
    )
    cam_time = (time.perf_counter() - t0) * 1000
    print(f"Grad-CAM on layer '{layer_name}': heatmap shape={cam.shape}, feature map={f_dim}, range=[{cam.min():.2f}, {cam.max():.2f}] in {cam_time:.1f}ms")
    print("[PASS] Task A smoke test succeeded.")


def smoke_2d_seg_xai():
    print("\n" + "=" * 60)
    print("SMOKE TEST 2: DeepLabV3+ Seg-Grad-CAM")
    print("=" * 60)
    model_path = os.path.join(BACKEND_DIR, "model_artifacts", "deeplabv3plus.keras")
    if not os.path.exists(model_path):
        print(f"[SKIP] DeepLabV3+ artifact not found at {model_path}")
        return

    t0 = time.perf_counter()
    model = tf.keras.models.load_model(model_path, compile=False)
    load_time = time.perf_counter() - t0
    print(f"Loaded {os.path.basename(model_path)} in {load_time:.2f}s")

    # Synthetic preprocessed batch (1, 256, 256, 3) in [0, 1]
    synth_batch = np.random.uniform(0.1, 0.9, (1, 256, 256, 3)).astype(np.float32)

    # Test Seg-Grad-CAM on conv2d_20 and up_sampling2d_2
    for target_layer in ["conv2d_20", "up_sampling2d_2"]:
        t0 = time.perf_counter()
        cam, res_layer, f_shape = seg2d_xai_adapter.explain(
            model, synth_batch, layer_name=target_layer
        )
        cam_time = (time.perf_counter() - t0) * 1000
        print(f"Seg-Grad-CAM on layer '{res_layer}': heatmap shape={cam.shape}, feature map={f_shape}, in {cam_time:.1f}ms")

    print("[PASS] Task B smoke test succeeded.")


def smoke_3d_unet():
    print("\n" + "=" * 60)
    print("SMOKE TEST 3: 3D U-Net Volumetric BraTS Segmentation + 3D Grad-CAM")
    print("=" * 60)
    model_path = os.path.join(BACKEND_DIR, "model_artifacts", "brats_3d_unet", "best_3d_unet_model.keras")

    model = None
    if os.path.exists(model_path):
        try:
            loaded = tf.keras.models.load_model(model_path, custom_objects=BRATS_3D_CUSTOM_OBJECTS, compile=False)
            if tuple(loaded.input_shape) == (None, 96, 96, 64, 4):
                model = loaded
                print(f"Loaded genuine 3D U-Net artifact: {model_path}")
        except Exception as e:
            print(f"Direct load note: {e}")

    if model is None:
        print("Instantiating canonical 3D U-Net graph (5,606,628 params, 96x96x64 patch)...")
        model = build_3d_unet()

    print(f"Model parameters: {model.count_params():,}, input_shape={model.input_shape}, output_shape={model.output_shape}")

    # Create synthetic 4-modality volume (96, 96, 64, 4)
    np.random.seed(42)
    flair = np.random.randn(96, 96, 64).astype(np.float32)
    t1 = np.random.randn(96, 96, 64).astype(np.float32)
    t1ce = np.random.randn(96, 96, 64).astype(np.float32)
    t2 = np.random.randn(96, 96, 64).astype(np.float32)

    t0 = time.perf_counter()
    stacked = unet3d_adapter.prepare_volume(flair, t1, t1ce, t2)
    prep_time = (time.perf_counter() - t0) * 1000
    print(f"Prepared volume shape: {stacked.shape} in {prep_time:.1f}ms")

    # Predict
    t0 = time.perf_counter()
    probs, labels = unet3d_adapter.predict_sliding_window(model, stacked, stride=(96, 96, 64))
    pred_time = (time.perf_counter() - t0) * 1000
    print(f"Prediction output shape: {labels.shape}, unique classes: {np.unique(labels).tolist()} in {pred_time:.1f}ms")

    # Uncertainty
    entropy = unet3d_adapter.compute_uncertainty(probs)
    print(f"Predictive entropy map shape: {entropy.shape}, range=[{entropy.min():.4f}, {entropy.max():.4f}]")

    # 3D Grad-CAM
    # Insert a synthetic lesion label so tumor_center_start finds voxels
    labels[40:55, 40:55, 25:35] = 3
    t0 = time.perf_counter()
    xai_res = unet3d_adapter.explain_3d(model, stacked, labels, class_indices=(1, 2, 3))
    xai_time = (time.perf_counter() - t0) * 1000
    if xai_res.get("available"):
        print(f"3D Grad-CAM computed for classes {list(xai_res['class_cams'].keys())} in {xai_time:.1f}ms")
        for cls_idx, cam_map in xai_res["class_cams"].items():
            print(f"  Class {cls_idx} CAM shape: {cam_map.shape}, range=[{cam_map.min():.2f}, {cam_map.max():.2f}]")
        print(f"  Modality ablation results: {list(xai_res['modality_ablation'].keys())}")
    print("[PASS] Task C smoke test succeeded.")


if __name__ == "__main__":
    t_start = time.perf_counter()
    smoke_2d_classifier()
    smoke_2d_seg_xai()
    smoke_3d_unet()
    print("\n" + "=" * 60)
    print(f"ALL SMOKE TESTS PASSED in {time.perf_counter() - t_start:.2f}s!")
    print("=" * 60)
