import numpy as np
from typing import Dict, Any, List, Optional
from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import ModelUnavailableException, InvalidConfigurationException
from app.ml.model_loader import model_loader
from app.ml.inference_utils import (
    make_gradcam,
    make_gradcam_plus_plus,
    make_segmentation_gradcam,
    compute_synthetic_heatmap_for_dev
)
from app.services.preprocessing_service import preprocessing_service
from app.services.imagekit_service import imagekit_service
from app.schemas.analysis import ExplanationResultItem

SUPPORTED_CLASSIFICATION_XAI = ["gradcam", "gradcam_pp"]
SUPPORTED_SEGMENTATION_XAI = ["seg_gradcam"]


class ExplainabilityService:
    async def generate_explanation(
        self,
        image_bytes: bytes,
        analysis_id: str,
        method: str = "gradcam",
        model_id: str = "efficientnetv2b0",
        target_layer: Optional[str] = None,
        target_class: Optional[str] = None,
        target_region: Optional[str] = "tumor_lesion"
    ) -> Dict[str, Any]:
        """
        Compute genuine model-derived explainability heatmap and overlay.
        Matches notebook implementation:
        - Grad-CAM on candidate layer 'top_conv'
        - Grad-CAM++ with higher-order gradients; reports failure accurately if unsupported
        - Genuine Segmentation XAI targeting foreground delineation
        - Overlays aligned and blended with original MRI
        - Uploaded to ImageKit with full audit metadata
        """
        if model_id == "attention_vit_segmentation":
            raise ModelUnavailableException(
                model_id,
                "The matching training preprocessing and output-channel mapping have not been verified; explainability is disabled",
            )

        method = method.lower()
        folder = f"neurolens/analyses/{analysis_id}"
        resolved_layer = target_layer or ("top_conv" if "seg" not in method else None)

        feature_map_str = None
        disclaimer_str = "Research/educational use only. Not clinically validated."

        # ── 1. CLASSIFICATION XAI ──────────────────────────────────────────
        if method in SUPPORTED_CLASSIFICATION_XAI:
            if model_loader.has_classification_model_for(model_id):
                if model_id == "swin_transformer" and method != "gradcam":
                    raise InvalidConfigurationException("Swin Transformer supports validated transformer Grad-CAM only.")
                model = model_loader.get_classification_model(model_id)
                tf = model_loader.get_tf()

                # Class index resolution
                class_names = [c.lower() for c in model_loader.get_class_names(model_id)]
                class_idx = None
                if target_class:
                    tc_clean = target_class.lower().replace(" ", "_")
                    if tc_clean in class_names:
                        class_idx = class_names.index(tc_clean)

                if model_id == "efficientnetv2b0_256":
                    from app.ml.adapters.efficientnet256_adapter import efficientnet256_adapter
                    x_256 = efficientnet256_adapter.preprocess_image(image_bytes)
                    target_lyr = resolved_layer or "top_activation"
                    heatmap_2d, resolved_layer, feat_dim = efficientnet256_adapter.explain_gradcam(
                        model, x_256, target_class_index=class_idx, target_layer=target_lyr
                    )
                    feature_map_str = f"{feat_dim[0]} × {feat_dim[1]} × {feat_dim[2]}"
                    pred_idx = class_idx if class_idx is not None else int(np.argmax(model(x_256[None], training=False).numpy()[0]))
                    resolved_class = model_loader.get_class_names(model_id)[pred_idx].replace("_", " ").title()
                    is_mock = False
                else:
                    cls_batch = (preprocessing_service.preprocess_for_swin_transformer(image_bytes)
                                 if model_id == "swin_transformer"
                                 else preprocessing_service.preprocess_for_classifier(image_bytes))

                    layer_to_use = ("swin_transformer_block_7" if model_id == "swin_transformer" and resolved_layer in (None, "top_conv")
                                    else (resolved_layer or "top_conv"))

                    try:
                        if method == "gradcam":
                            heatmap_2d, pred_idx, _ = make_gradcam(
                                model, cls_batch, class_index=class_idx,
                                conv_layer_name=layer_to_use, tf_module=tf
                            )
                        else:  # gradcam_pp
                            heatmap_2d, pred_idx, _ = make_gradcam_plus_plus(
                                model, cls_batch, class_index=class_idx,
                                conv_layer_name=layer_to_use, tf_module=tf
                            )
                        is_mock = False
                        resolved_class = model_loader.get_class_names(model_id)[pred_idx].replace("_", " ").title()
                        feature_map_str = "7 × 7 × 768" if model_id == "swin_transformer" else "8 × 8"

                    except Exception as e:
                        logger.error(f"XAI calculation failed for {method} on {layer_to_use}: {e}")
                        raise InvalidConfigurationException(
                            f"Explanation computation failed for {method} on layer '{layer_to_use}': {str(e)}"
                        )

            else:
                if settings.is_production:
                    raise ModelUnavailableException(model_id, f"Classification model missing for {method}")
                if not settings.USE_MOCK_MODELS_IN_DEV:
                    raise ModelUnavailableException(model_id, "Model missing and mock dev mode disabled")

                seed = 1 if method == "gradcam" else 2
                heatmap_2d = compute_synthetic_heatmap_for_dev(settings.IMG_SIZE, settings.IMG_SIZE, seed=seed)
                resolved_class = target_class or "Glioma"
                resolved_layer = resolved_layer or "top_conv"
                feature_map_str = "8 × 8"
                is_mock = True

        # ── 2. SEGMENTATION XAI ────────────────────────────────────────────
        elif method in SUPPORTED_SEGMENTATION_XAI:
            if model_loader.has_segmentation_model_for(model_id):
                seg_model = model_loader.get_segmentation_model(model_id)
                tf = model_loader.get_tf()
                seg_batch = (preprocessing_service.preprocess_for_deeplabv3plus(image_bytes)
                             if model_id == "deeplabv3plus"
                             else preprocessing_service.preprocess_for_segmenter(image_bytes))

                try:
                    if model_id == "deeplabv3plus":
                        from app.ml.adapters.seg2d_xai_adapter import seg2d_xai_adapter
                        heatmap_2d, resolved_layer, feat_dim = seg2d_xai_adapter.explain(
                            seg_model, seg_batch, layer_name=resolved_layer
                        )
                        feature_map_str = f"{feat_dim[0]} × {feat_dim[1]} × {feat_dim[2]}"
                        resolved_class = "Tumor Lesion Area"
                    else:
                        heatmap_2d = make_segmentation_gradcam(
                            seg_model, seg_batch, conv_layer_name=resolved_layer, tf_module=tf
                        )
                        resolved_class = "Tumor Lesion Boundary"
                        resolved_layer = resolved_layer or "bottleneck_conv"
                        feature_map_str = "16 × 16"
                    is_mock = False
                except Exception as e:
                    logger.error(f"Segmentation XAI failed: {e}")
                    raise InvalidConfigurationException(f"Segmentation XAI computation failed: {str(e)}")
            else:
                if settings.is_production:
                    raise ModelUnavailableException("unet", "Segmentation model missing for Seg-GradCAM")
                if not settings.USE_MOCK_MODELS_IN_DEV:
                    raise ModelUnavailableException("unet", "Model missing and mock dev mode disabled")

                heatmap_2d = compute_synthetic_heatmap_for_dev(settings.IMG_SIZE, settings.IMG_SIZE, seed=3)
                resolved_class = "Tumor Lesion Boundary"
                resolved_layer = resolved_layer or "bottleneck_conv"
                feature_map_str = "16 × 16"
                is_mock = True

        else:
            raise InvalidConfigurationException(
                f"Unsupported XAI method '{method}'. Supported methods: {SUPPORTED_CLASSIFICATION_XAI + SUPPORTED_SEGMENTATION_XAI}"
            )

        # Generate Jet Heatmap and Blended Overlay PNG bytes
        heatmap_bytes, overlay_bytes = preprocessing_service.overlay_heatmap_on_image(
            image_bytes, heatmap_2d, alpha=0.40
        )

        # Upload artifacts to ImageKit
        heatmap_upload = await imagekit_service.upload_file(
            heatmap_bytes, f"{method}_heatmap.png", folder=folder
        )
        overlay_upload = await imagekit_service.upload_file(
            overlay_bytes, f"{method}_overlay.png", folder=folder
        )

        descriptions = {
            "gradcam": (
                f"Gradient-weighted Class Activation Map from layer '{resolved_layer}' "
                f"visualizing visual features contributing to '{resolved_class}' prediction."
                if model_id != "efficientnetv2b0_256" else
                f"Gradient-weighted Class Activation Map from layer '{resolved_layer}' (feature map {feature_map_str or '8x8x1280'}) "
                f"visualizing regions that influenced the '{resolved_class}' prediction. Saliency highlights coarse regions."
            ),
            "gradcam_pp": (
                f"Grad-CAM++ higher-order saliency map from layer '{resolved_layer}' "
                f"providing fine-grained spatial attribution for '{resolved_class}'."
            ),
            "seg_gradcam": (
                "Shows which features drove the total predicted tumor area."
                if model_id == "deeplabv3plus" else
                f"Segmentation-specific attribution backpropagated from foreground lesion output through latent layer '{resolved_layer}'."
            )
        }

        desc = descriptions.get(method, f"Attribution map for {resolved_class} on {resolved_layer}.")

        result_item = ExplanationResultItem(
            method=method,
            heatmapDataUrl=heatmap_upload["url"],
            overlayDataUrl=overlay_upload["url"],
            targetClass=resolved_class,
            layerName=resolved_layer,
            description=desc,
            featureMapShape=feature_map_str,
            disclaimer=disclaimer_str,
        )

        return {
            "result_item": result_item,
            "method": method,
            "target_layer": resolved_layer,
            "target_class": resolved_class,
            "target_region": target_region,
            "heatmap_url": heatmap_upload["url"],
            "overlay_url": overlay_upload["url"],
            "description": desc,
            "feature_map_shape": feature_map_str,
            "disclaimer": disclaimer_str,
            "is_mock": is_mock
        }


explainability_service = ExplainabilityService()
