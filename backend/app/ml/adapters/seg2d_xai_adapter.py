"""
Adapter for 2D DeepLabV3+ Seg-Grad-CAM explainability.
Model Artifact: deeplabv3plus.keras
Input: (None, 256, 256, 3) float32 RGB [0, 1]
Output: (None, 256, 256, 1) sigmoid probability map
Target Score: Sum of predicted tumor probability over all pixels.
Verified Layers:
  - conv2d_20 (ASPP fused output activation: 16x16x256)
  - up_sampling2d_2 (Last decoder activation before final x4 upsampling: 64x64x256)
"""

import io
import numpy as np
from typing import Dict, Any, Tuple, Optional
from PIL import Image

from app.ml.seg2d_xai import (
    resolve_layers as ref_resolve_layers,
    feature_map_shape as ref_feature_map_shape,
    seg_gradcam as ref_seg_gradcam,
    NOTEBOOK_LAYERS,
)

METADATA = {
    "model_id": "deeplabv3plus",
    "display_name": "DeepLabV3+ (ResNet50 backbone)",
    "task": "2D Binary Lesion Segmentation",
    "xai_method": "Seg-Grad-CAM",
    "caption": "Shows which features drove the total predicted tumor area.",
    "disclaimer": "Research/educational use only. Not clinically validated.",
    "default_layer": "conv2d_20",
    "candidate_layers": [
        {
            "id": "conv2d_20",
            "name": "conv2d_20",
            "role": "ASPP Fused Output Activation",
            "feature_map": "16 × 16 × 256",
            "description": "Deep multiscale context from Atrous Spatial Pyramid Pooling",
        },
        {
            "id": "up_sampling2d_2",
            "name": "up_sampling2d_2",
            "role": "Decoder Feature Map",
            "feature_map": "64 × 64 × 256",
            "description": "High-resolution boundary delineation features before final x4 upsampling",
        },
    ],
    "test_dice": 0.848,
    "test_iou": 0.754,
    "evaluation_split": "LGG MRI segmentation dataset, patient-grouped split",
}


class Seg2DXaiAdapter:
    @staticmethod
    def get_available_layers(model: Any) -> Dict[str, Any]:
        """Resolves verified layers present in the loaded DeepLabV3+ model."""
        return ref_resolve_layers(model, img_size=256)

    @staticmethod
    def get_feature_map_shape(model: Any, layer_name: str) -> Tuple[int, int, int]:
        return ref_feature_map_shape(model, layer_name)

    @staticmethod
    def explain(
        model: Any,
        x_batch: np.ndarray,
        layer_name: Optional[str] = None,
        out_size: Tuple[int, int] = (256, 256),
    ) -> Tuple[np.ndarray, str, Tuple[int, int, int]]:
        """
        Computes Seg-Grad-CAM attribution map.
        x_batch: (1, 256, 256, 3) float32 in [0, 1].
        Returns:
          - heatmap: float32 (H, W) in [0, 1]
          - resolved_layer: layer name used
          - shape: (h, w, c) of the layer feature map
        """
        # Resolve target layer
        resolved = layer_name
        if not resolved:
            layers_info = ref_resolve_layers(model, img_size=256)
            candidates = layers_info.get("layers", [])
            resolved = "conv2d_20" if "conv2d_20" in candidates else (candidates[0] if candidates else "conv2d_20")

        heatmap = ref_seg_gradcam(model, x_batch, resolved, out_size=out_size)
        f_shape = ref_feature_map_shape(model, resolved)
        return heatmap, resolved, f_shape


seg2d_xai_adapter = Seg2DXaiAdapter()
