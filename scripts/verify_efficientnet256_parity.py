#!/usr/bin/env python3
"""
scripts/verify_efficientnet256_parity.py

Compares backend EfficientNet256Adapter preprocessing against reference implementation
(with simulate_training_jpeg=True matching the training pipeline).
Reports:
  - Max absolute difference in class probabilities across N samples.
  - Any predicted class flips.
"""

import os
import sys
import glob
import numpy as np
import tensorflow as tf

# Ensure backend and root are on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from backend.app.ml.adapters.efficientnet256_adapter import efficientnet256_adapter
from reference.classifier_efficientnet256 import preprocess as ref_preprocess, decode_image_bytes, predict as ref_predict


def find_sample_images(n_samples: int = 10):
    upload_dir = os.path.join(BACKEND_DIR, "storage", "uploads")
    patterns = [os.path.join(upload_dir, "*.png"), os.path.join(upload_dir, "*.jpg")]
    found = []
    for pat in patterns:
        found.extend(glob.glob(pat))

    if len(found) >= n_samples:
        return found[:n_samples]

    # If insufficient samples found, generate synthetic MRI-like samples
    import cv2
    synthetic_samples = []
    temp_dir = os.path.join(BASE_DIR, "tests", "samples", "temp_mri")
    os.makedirs(temp_dir, exist_ok=True)
    needed = n_samples - len(found)
    for i in range(needed):
        synth = np.zeros((256, 256, 3), dtype=np.uint8)
        # Draw brain ellipse and tumor-like circle
        cv2.ellipse(synth, (128, 128), (90, 110), 0, 0, 360, (180, 180, 180), -1)
        cv2.circle(synth, (128 + (i * 5) % 40, 128 + (i * 7) % 40), 20 + i % 15, (240, 240, 240), -1)
        # Add texture
        noise = np.random.randint(0, 30, (256, 256, 3), dtype=np.uint8)
        synth = cv2.add(synth, noise)
        p = os.path.join(temp_dir, f"synth_{i}.png")
        cv2.imwrite(p, synth)
        synthetic_samples.append(p)
    return (found + synthetic_samples)[:n_samples]


def run_parity_check(n_samples: int = 10):
    model_path = os.path.join(BACKEND_DIR, "model_artifacts", "EfficientNetV2B0_clahe_256.keras")
    if not os.path.exists(model_path):
        print(f"[ERROR] Artifact not found at: {model_path}")
        sys.exit(1)

    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path, compile=False)

    image_paths = find_sample_images(n_samples)
    print(f"Evaluating {len(image_paths)} sample images for preprocessing parity...")

    max_prob_diff = 0.0
    flips = 0

    for idx, path in enumerate(image_paths):
        with open(path, "rb") as f:
            raw_bytes = f.read()

        # 1. Backend adapter preprocessing (production path)
        x_backend = efficientnet256_adapter.preprocess_image(raw_bytes, simulate_training_jpeg=False)

        # 2. Reference preprocessing (with simulate_training_jpeg=True)
        img_bgr = decode_image_bytes(raw_bytes)
        x_ref = ref_preprocess(img_bgr, simulate_training_jpeg=True)

        # Predict
        p_backend = ref_predict(model, x_backend)
        p_ref = ref_predict(model, x_ref)

        diff = float(np.max(np.abs(p_backend - p_ref)))
        if diff > max_prob_diff:
            max_prob_diff = diff

        top_backend = int(np.argmax(p_backend))
        top_ref = int(np.argmax(p_ref))
        if top_backend != top_ref:
            flips += 1
            print(f"  [FLIP] Sample {idx} ({os.path.basename(path)}): backend={top_backend} vs ref={top_ref} (diff={diff:.4f})")

    print("\n" + "=" * 60)
    print(f"Parity Check Summary ({len(image_paths)} samples):")
    print(f"  Max Absolute Probability Difference: {max_prob_diff:.6f}")
    print(f"  Predicted Class Flips: {flips} / {len(image_paths)}")
    print("=" * 60)

    if flips > 0:
        print("[WARNING] Class flips detected between production and training JPEG round-trip.")
    else:
        print("[SUCCESS] Zero class flips. Preprocessing is highly consistent.")
    return max_prob_diff, flips


if __name__ == "__main__":
    run_parity_check(10)
