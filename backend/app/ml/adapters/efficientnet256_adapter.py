"""
Adapter for EfficientNetV2B0 256x256 CLAHE Classifier.
Model Artifact: EfficientNetV2B0_clahe_256.keras
Input: (None, 256, 256, 3) float32 RAW [0, 255] (Model contains internal Rescaling/Normalization)
Output: 4 classes: [0: glioma, 1: meningioma, 2: notumor, 3: pituitary]
Layer: top_activation (8x8x1280)
"""

import io
import cv2
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Tuple, Optional
from PIL import Image

# Import verified math from internal app.ml module
from app.ml.classifier_efficientnet256 import (
    CLASS_NAMES as REF_CLASS_NAMES,
    INPUT_SIZE,
    GRADCAM_LAYER,
    clahe_gray3,
    preprocess as ref_preprocess,
    predict as ref_predict,
    gradcam as ref_gradcam,
    feature_map_shape as ref_feature_map_shape,
)

# Standard app label mapping
APP_CLASS_MAPPING = {
    0: "Glioma",
    1: "Meningioma",
    2: "No Tumor",
    3: "Pituitary",
}

# Verified metadata for provenance and UI display
METADATA = {
    "model_id": "efficientnetv2b0_256",
    "model_name": "EfficientNetV2B0 (CLAHE, 256x256)",
    "model_version": "v2.5.0-clahe256",
    "architecture": "EfficientNetV2-B0 (Fused-MBConv + Squeeze-and-Excitation)",
    "input_resolution": "256 × 256 × 3 RGB",
    "parameters_count": "7.11M params",
    "target_layer": "top_activation",
    "feature_map_shape": "8 × 8 × 1280",
    "preprocessing": "Grayscale CLAHE (clip 2.0, grid 8x8) -> 3-ch RGB -> Bilinear 256x256 -> Float32 [0, 255]",
    "test_accuracy": 0.9625,
    "evaluation_split": "1,440-image held-out split (random per-image split, no patient IDs, no external validation)",
    "saliency_disclaimer": "Research/educational use only. Not clinically validated. Saliency highlights regions that influenced the classification and are coarse; they do not delineate clinical tumor boundaries.",
}


class EfficientNet256Adapter:
    @staticmethod
    def preprocess_image(image_bytes: bytes, simulate_training_jpeg: bool = False) -> np.ndarray:
        """
        Processes raw image bytes into tensor shape (256, 256, 3) in RAW 0-255 range.
        Applies CLAHE at native resolution first, then bilinearly resizes to 256x256.
        """
        arr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            # Fallback via PIL
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_rgb = np.asarray(pil_img)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        return ref_preprocess(img_bgr, simulate_training_jpeg=simulate_training_jpeg)

    @staticmethod
    def predict(model: Any, x_256: np.ndarray) -> Dict[str, Any]:
        """
        Inference step on preprocessed (256, 256, 3) tensor.
        """
        probs = ref_predict(model, x_256)
        pred_idx = int(np.argmax(probs))
        pred_raw = REF_CLASS_NAMES[pred_idx]
        display_label = APP_CLASS_MAPPING[pred_idx]

        score_dict = {
            APP_CLASS_MAPPING[i]: float(probs[i]) for i in range(len(REF_CLASS_NAMES))
        }

        return {
            "predicted_index": pred_idx,
            "predicted_class": display_label,
            "predicted_class_raw": pred_raw,
            "confidence": float(probs[pred_idx]),
            "probabilities": score_dict,
            "raw_scores": [float(p) for p in probs],
        }

    @staticmethod
    def explain_gradcam(
        model: Any,
        x_256: np.ndarray,
        target_class_index: Optional[int] = None,
        target_layer: str = GRADCAM_LAYER,
        out_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, str, Tuple[int, int, int]]:
        """
        Computes model-derived Grad-CAM for top_activation (8x8x1280).
        Returns:
          - cam_2d: normalized float32 heatmap [0, 1] resized to out_size or (256, 256)
          - layer_name: "top_activation"
          - feature_map_dim: (8, 8, 1280)
        """
        cam_2d, resolved_idx, _ = ref_gradcam(
            model,
            x_256,
            layer_name=target_layer,
            class_index=target_class_index,
            out_size=out_size or (256, 256),
        )
        f_shape = ref_feature_map_shape(model, target_layer)
        return cam_2d, target_layer, f_shape


efficientnet256_adapter = EfficientNet256Adapter()
