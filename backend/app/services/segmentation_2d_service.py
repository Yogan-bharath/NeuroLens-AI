"""
2D Tumor Segmentation Service
Runs a loaded binary segmentation model (BRISC U-Net or DeepLabV3+) on a 2D MRI slice
to produce a genuine predicted tumor mask and overlay, independent of Grad-CAM.
"""

import io
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional

from app.core.config import settings
from app.core.logging import logger
from app.ml.model_loader import model_loader
from app.services.preprocessing_service import preprocessing_service
from app.services.imagekit_service import imagekit_service


# Color for tumor region overlay (red channel highlighted)
TUMOR_MASK_COLOR = (239, 68, 68)  # Red #EF4444
BOUNDARY_COLOR = (245, 158, 11)   # Amber #F59E0B


def _render_mask_rgba(
    original_rgb: np.ndarray,
    binary_mask: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Overlay a binary tumor mask on top of the original MRI image.
    Tumor pixels are highlighted in red.
    Returns uint8 RGBA image array.
    """
    h, w = original_rgb.shape[:2]

    # Ensure mask dimensions match
    if binary_mask.shape != (h, w):
        mask_pil = Image.fromarray((binary_mask.astype(np.uint8) * 255))
        mask_pil = mask_pil.resize((w, h), Image.Resampling.NEAREST)
        binary_mask = np.asarray(mask_pil) > 0

    out = original_rgb.astype(np.float32) / 255.0

    # Red tumor highlight
    tumor_locs = binary_mask > 0
    if np.any(tumor_locs):
        r, g, b = TUMOR_MASK_COLOR
        out[tumor_locs, 0] = out[tumor_locs, 0] * (1.0 - alpha) + (r / 255.0) * alpha
        out[tumor_locs, 1] = out[tumor_locs, 1] * (1.0 - alpha) + (g / 255.0) * alpha
        out[tumor_locs, 2] = out[tumor_locs, 2] * (1.0 - alpha) + (b / 255.0) * alpha

    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out


def _render_mask_only(binary_mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Render a clean mask-only image: tumor in red on black background.
    Returns uint8 RGB array of shape (h, w, 3).
    """
    if binary_mask.shape != (h, w):
        mask_pil = Image.fromarray((binary_mask.astype(np.uint8) * 255))
        mask_pil = mask_pil.resize((w, h), Image.Resampling.NEAREST)
        binary_mask = np.asarray(mask_pil) > 0

    out = np.zeros((h, w, 3), dtype=np.uint8)
    tumor_locs = binary_mask > 0
    if np.any(tumor_locs):
        r, g, b = TUMOR_MASK_COLOR
        out[tumor_locs, 0] = r
        out[tumor_locs, 1] = g
        out[tumor_locs, 2] = b
    return out


def _add_boundary_outline(
    overlay_rgb: np.ndarray,
    binary_mask: np.ndarray,
) -> np.ndarray:
    """
    Add a thin boundary contour around the predicted tumor region.
    """
    import cv2
    h, w = overlay_rgb.shape[:2]
    if binary_mask.shape != (h, w):
        mask_pil = Image.fromarray((binary_mask.astype(np.uint8) * 255))
        mask_pil = mask_pil.resize((w, h), Image.Resampling.NEAREST)
        binary_mask = np.asarray(mask_pil) > 0

    mask_uint8 = (binary_mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        r, g, b = BOUNDARY_COLOR
        cv2.drawContours(overlay_rgb, contours, -1, (int(r), int(g), int(b)), 2)
    return overlay_rgb


class Segmentation2DService:
    """
    Runs a loaded binary segmentation model on a 2D MRI to produce a real
    predicted tumor mask. This is NOT Grad-CAM — it is model.predict().
    """

    def _select_model(self):
        """
        Try to use DeepLabV3+ first (better Dice), fall back to BRISC U-Net.
        Returns (model, model_id, preprocess_fn).
        """
        if model_loader.has_segmentation_model_for("deeplabv3plus"):
            return (
                model_loader.get_segmentation_model("deeplabv3plus"),
                "deeplabv3plus",
                preprocessing_service.preprocess_for_deeplabv3plus,
            )
        if model_loader.has_segmentation_model_for("unet"):
            return (
                model_loader.get_segmentation_model("unet"),
                "unet",
                preprocessing_service.preprocess_for_segmenter,
            )
        return None, None, None

    async def segment_2d(
        self,
        image_bytes: bytes,
        analysis_id: str,
        threshold: float = 0.5,
        min_area_pixels: int = 50,
        explain: bool = False,
        explain_layer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run actual 2D segmentation inference.
        1. Preprocess image for the available segmentation model.
        2. model.predict() → raw probability map.
        3. Threshold to binary mask.
        4. Filter small noise regions (< min_area_pixels).
        5. Render overlay and pure mask images.
        6. Upload to ImageKit.
        Returns dict with mask_url, overlay_url, model_used, tumor_detected, tumor_area_pct.
        """
        model, model_id, preprocess_fn = self._select_model()

        if model is None:
            raise RuntimeError(
                "No 2D segmentation model is loaded (neither DeepLabV3+ nor BRISC U-Net). "
                "Cannot produce tumor segmentation mask."
            )

        # 1. Preprocess
        input_batch = preprocess_fn(image_bytes)

        # 2. Inference — genuine model prediction
        raw_output = model(input_batch, training=False).numpy()[0]

        # Handle different output shapes
        if raw_output.ndim == 3 and raw_output.shape[-1] == 1:
            # Binary segmentation (BRISC U-Net): (H, W, 1) sigmoid
            prob_map = raw_output.squeeze(-1)
        elif raw_output.ndim == 3 and raw_output.shape[-1] == 2:
            # Two-class segmentation: take class 1 (tumor)
            prob_map = raw_output[:, :, 1]
        elif raw_output.ndim == 2:
            prob_map = raw_output
        else:
            # Multi-class: sum all non-background classes
            prob_map = 1.0 - raw_output[:, :, 0]

        # 3. Threshold
        binary_mask = (prob_map >= threshold).astype(np.uint8)

        # 4. Remove small connected components (noise filtering)
        try:
            from scipy.ndimage import label as scipy_label
            labeled, num_features = scipy_label(binary_mask)
            for comp_idx in range(1, num_features + 1):
                comp_area = int(np.sum(labeled == comp_idx))
                if comp_area < min_area_pixels:
                    binary_mask[labeled == comp_idx] = 0
        except ImportError:
            pass

        # 5. Compute tumor statistics
        tumor_pixel_count = int(np.sum(binary_mask > 0))
        total_pixels = int(binary_mask.shape[0] * binary_mask.shape[1])
        tumor_area_pct = round((tumor_pixel_count / total_pixels) * 100.0, 2) if total_pixels > 0 else 0.0
        tumor_detected = tumor_pixel_count > 0

        # 6. Render images
        pil_rgb = preprocessing_service.load_pil_rgb(image_bytes)
        original_rgb = np.asarray(pil_rgb)

        # Overlay: original MRI + semi-transparent red tumor highlight + boundary contour
        overlay_rgb = _render_mask_rgba(original_rgb, binary_mask, alpha=0.45)
        overlay_rgb = _add_boundary_outline(overlay_rgb, binary_mask)

        # Pure mask image
        mask_only_rgb = _render_mask_only(binary_mask, original_rgb.shape[0], original_rgb.shape[1])

        # Encode to PNG bytes
        overlay_buf = io.BytesIO()
        Image.fromarray(overlay_rgb).save(overlay_buf, format="PNG")
        overlay_bytes = overlay_buf.getvalue()

        mask_buf = io.BytesIO()
        Image.fromarray(mask_only_rgb).save(mask_buf, format="PNG")
        mask_bytes = mask_buf.getvalue()

        # 7. Upload to ImageKit
        folder = f"neurolens/analyses/{analysis_id}"
        mask_upload = await imagekit_service.upload_file(
            mask_bytes, "segmentation_mask.png", folder=folder
        )
        overlay_upload = await imagekit_service.upload_file(
            overlay_bytes, "segmentation_overlay.png", folder=folder
        )

        explain_data: Dict[str, Any] = {}
        if explain:
            try:
                import cv2
                from app.ml.seg2d_explain import generate_segmentation_explanation
                target_lyr = explain_layer or "up_sampling2d_2"
                h, w = original_rgb.shape[:2]
                # Two separate conversions:
                # 1) DeepLab model operates on 256x256 input_batch
                # 2) Rescale mask & CAM heatmap to the native MRI image size (h, w)
                mask_native = cv2.resize(
                    binary_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                )
                exp_res = generate_segmentation_explanation(
                    model=model,
                    x_batch=input_batch,
                    out_size=(h, w),
                    pred_mask=mask_native,
                    preferred_layer=target_lyr,
                )
                
                exp_url = None
                if exp_res.get("available") and exp_res.get("heatmap_png_bytes"):
                    try:
                        exp_upload = await imagekit_service.upload_file(
                            exp_res["heatmap_png_bytes"], "segmentation_heatmap_rgba.png", folder=folder
                        )
                        exp_url = exp_upload.get("url")
                    except Exception as up_err:
                        logger.warning(f"Failed to upload heatmap to ImageKit: {up_err}")
                
                # Use uploaded URL or base64 data URL
                effective_heatmap = exp_url or exp_res.get("heatmap_png")
                
                explain_payload = {
                    "available": exp_res.get("available", False),
                    "heatmap_png": effective_heatmap,
                    "layer": exp_res.get("layer", target_lyr),
                    "map_shape": exp_res.get("map_shape", "16 × 16 × 256"),
                    "stats": exp_res.get("stats", {}),
                    "method": exp_res.get("method", "SEG-GRADCAM"),
                    "caption": exp_res.get("caption", "Shows which features drove the total predicted tumor area."),
                    "disclaimer": exp_res.get("disclaimer", "Research/educational use only. Not clinically validated."),
                }
                if not exp_res.get("available") and "reason" in exp_res:
                    explain_payload["reason"] = exp_res["reason"]

                explain_data = {
                    "explain": explain_payload,
                    "explain_url": effective_heatmap,
                    "explain_layer": exp_res.get("layer", target_lyr),
                    "explain_feature_map": exp_res.get("map_shape", "256 × 256 × 256"),
                    "explain_caption": exp_res.get("caption", "Shows which features drove the total predicted tumor area."),
                    "explain_heatmap_png": effective_heatmap,
                    "explain_stats": exp_res.get("stats", {}),
                    "explain_available": exp_res.get("available", False),
                }
            except Exception as exp_err:
                logger.warning(f"Seg-Grad-CAM generation failed: {exp_err}")
                explain_data = {
                    "explain": {
                        "available": False,
                        "reason": f"Seg-Grad-CAM generation error: {exp_err}",
                        "method": "SEG-GRADCAM",
                        "heatmap_png": None,
                        "disclaimer": "Research/educational use only. Not clinically validated.",
                    },
                    "explain_available": False,
                }

        logger.info(
            "2D segmentation complete: model=%s, tumor_detected=%s, area=%.2f%%, pixels=%d",
            model_id, tumor_detected, tumor_area_pct, tumor_pixel_count,
        )

        result_dict = {
            "mask_url": mask_upload["url"],
            "overlay_url": overlay_upload["url"],
            "model_used": model_id,
            "tumor_detected": tumor_detected,
            "tumor_area_pct": tumor_area_pct,
            "tumor_pixel_count": tumor_pixel_count,
            "threshold": threshold,
        }
        result_dict.update(explain_data)
        return result_dict


segmentation_2d_service = Segmentation2DService()

