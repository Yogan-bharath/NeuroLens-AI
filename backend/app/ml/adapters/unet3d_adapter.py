"""
Adapter for 3D U-Net Volumetric BraTS Segmentation & XAI.
Model Artifact: best_3d_unet_model.keras (under backend/model_artifacts/brats_3d_unet/)
Input: (None, 96, 96, 64, 4) float32 (H, W, D, 4) in native NIfTI order [FLAIR, T1, T1CE, T2]
Output: (None, 96, 96, 64, 4) softmax over 4 classes:
  0: Background
  1: Necrotic / Non-enhancing
  2: Edema
  3: Enhancing tumor (BraTS label 4 mapped to 3)
Layer: bottleneck_relu2 (12x12x8x256)
"""

import os
import json
import numpy as np
import tensorflow as tf
from typing import Dict, Any, Tuple, Optional, Callable, Sequence
import nibabel as nib

from reference.unet3d import (
    PATCH_SHAPE,
    STRIDE,
    MODALITIES,
    NUM_CLASSES,
    CLASS_NAMES,
    GRADCAM_LAYER,
    CUSTOM_OBJECTS,
    normalize_modality as ref_normalize_modality,
    prepare_volume as ref_prepare_volume,
    sliding_window_predict as ref_sliding_window_predict,
    labels_from_probs as ref_labels_from_probs,
    prediction_entropy as ref_prediction_entropy,
    resize_trilinear as ref_resize_trilinear,
    gradcam3d as ref_gradcam3d,
    tumor_center_start as ref_tumor_center_start,
    extract_patch as ref_extract_patch,
    embed_patch_map as ref_embed_patch_map,
)

METADATA = {
    "model_id": "unet3d_brats",
    "display_name": "3D U-Net (volumetric)",
    "architecture": "Deep 3D Residual/Encoder-Decoder U-Net (Conv3D + BatchNorm)",
    "parameters_count": "5.61M params (5,606,628)",
    "input_spec": "96 × 96 × 64 × 4 patches (FLAIR, T1, T1CE, T2), sliding window stride 64×64×32",
    "evaluation": "BraTS 2020, patient-level split, complete-volume sliding-window evaluation on 56 held-out patients",
    "metrics": {
        "whole_tumor_dice": 0.829760,
        "whole_tumor_iou": 0.728292,
        "tumor_core_dice": 0.696069,
        "tumor_core_iou": 0.593072,
        "necrotic_dice": 0.470383,
        "necrotic_iou": 0.360244,
        "edema_dice": 0.694502,
        "edema_iou": 0.572507,
        "enhancing_dice": 0.641072,
        "enhancing_iou": 0.544092,
        "mean_foreground_dice": 0.601986,
        "mean_foreground_iou": 0.492281,
        "pixel_accuracy": 0.995746,
    },
    "comparison_note": "The 2D Attention CNN + ViT whole-tumor Dice of 0.8937 used selected tumor-enriched slices; the protocols differ, so the numbers are not a head-to-head comparison.",
    "gradcam": {
        "layer": "bottleneck_relu2",
        "feature_map": "12 × 12 × 8 × 256",
        "scope": "96 × 96 × 64 tumor-centred patch",
    },
    "banner": "GENUINE MODEL PREDICTION - 3D GRAD-CAM (BOTTLENECK, PATCH-LEVEL)",
    "disclaimer": "Research/educational use only. Not clinically validated.",
}


class UNet3DAdapter:
    @staticmethod
    def prepare_volume(flair: np.ndarray, t1: np.ndarray, t1ce: np.ndarray, t2: np.ndarray) -> np.ndarray:
        """Stacks and normalizes non-zero voxels across 4 modalities: shape (H, W, D, 4)."""
        return ref_prepare_volume(flair, t1, t1ce, t2)

    @staticmethod
    def predict_sliding_window(
        model: Any,
        stacked_volume: np.ndarray,
        stride: Tuple[int, int, int] = STRIDE,
        batch_size: int = 2,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs sliding-window inference with averaging over overlaps.
        Returns:
          - probs: float32 (H, W, D, 4)
          - labels: uint8 (H, W, D) in {0, 1, 2, 3}
        """
        probs = ref_sliding_window_predict(
            model,
            stacked_volume,
            patch_shape=PATCH_SHAPE,
            stride=stride,
            batch_size=batch_size,
            progress_cb=progress_cb,
            num_classes=NUM_CLASSES,
        )
        labels = ref_labels_from_probs(probs)
        return probs, labels

    @staticmethod
    def explain_3d(
        model: Any,
        stacked_volume: np.ndarray,
        pred_labels: np.ndarray,
        class_indices: Sequence[int] = (1, 2, 3),
    ) -> Dict[str, Any]:
        """
        Computes 3D Grad-CAM, bottleneck activations, modality ablation, and entropy uncertainty.
        Centers the 96x96x64 patch on the median of PREDICTED tumor voxels without ground truth.
        """
        start = ref_tumor_center_start(pred_labels, patch_shape=PATCH_SHAPE)
        if start is None:
            return {
                "available": False,
                "reason": "No tumor voxels predicted; 3D Grad-CAM is unavailable.",
            }

        patch = ref_extract_patch(stacked_volume, start, patch_shape=PATCH_SHAPE)

        # 1. 3D Grad-CAM with trilinear interpolation
        cams = ref_gradcam3d(model, patch, class_indices=class_indices, layer_name=GRADCAM_LAYER)

        # Embed patch heatmap into full volume grid
        embedded_maps = {}
        for c, cam_patch in cams.items():
            embedded_maps[c] = ref_embed_patch_map(cam_patch, start, stacked_volume.shape[:3])

        # 2. Bottleneck latent features
        try:
            layer = model.get_layer(GRADCAM_LAYER)
            feat_model = tf.keras.Model(model.inputs, layer.output)
            b_feats = feat_model(patch[None].astype(np.float32), training=False).numpy()[0]
            # Mean activation map over channels
            mean_b_act = np.mean(b_feats, axis=-1)
            # Top 3 channels by variance
            variances = np.var(b_feats, axis=(0, 1, 2))
            top3_channels = np.argsort(variances)[-3:][::-1].tolist()
        except Exception:
            mean_b_act = None
            top3_channels = []

        # 3. Modality ablation
        # Patch-level ablation: zero each modality and measure relative volume shift
        ablation_results = {}
        base_pred = np.argmax(model(patch[None].astype(np.float32), training=False).numpy()[0], axis=-1)
        base_tumor_voxels = int(np.sum(base_pred > 0))

        for idx, mod in enumerate(MODALITIES):
            ablated_patch = patch.copy()
            ablated_patch[..., idx] = 0.0
            abl_pred = np.argmax(model(ablated_patch[None].astype(np.float32), training=False).numpy()[0], axis=-1)
            abl_tumor_voxels = int(np.sum(abl_pred > 0))
            delta = abl_tumor_voxels - base_tumor_voxels
            pct_change = round((delta / max(base_tumor_voxels, 1)) * 100.0, 2)
            ablation_results[mod.upper()] = {
                "tumor_voxels": abl_tumor_voxels,
                "delta": delta,
                "percent_change": pct_change,
            }

        return {
            "available": True,
            "banner": METADATA["banner"],
            "patch_start": start,
            "patch_shape": PATCH_SHAPE,
            "layer_name": GRADCAM_LAYER,
            "feature_map_shape": "12 × 12 × 8 × 256",
            "class_cams": cams,
            "embedded_cams": embedded_maps,
            "top3_channels": top3_channels,
            "modality_ablation": ablation_results,
        }

    @staticmethod
    def compute_uncertainty(probs: np.ndarray) -> np.ndarray:
        """Normalized voxel-wise predictive entropy in [0, 1]."""
        return ref_prediction_entropy(probs)


unet3d_adapter = UNet3DAdapter()
