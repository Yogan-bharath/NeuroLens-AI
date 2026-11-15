import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.api.deps import get_current_user_or_default
from app.core.config import settings
from app.core.exceptions import InvalidConfigurationException, ModelUnavailableException
from app.core.logging import logger
from app.db.mongodb import get_database
from app.ml.model_loader import model_loader
from app.ml.model_registry import get_all_classification_models
from app.models.user import UserDB
from app.services.classification_service import classification_service
from app.services.explainability_service import explainability_service
from app.services.analysis_service import analysis_service, _in_memory_analyses
from app.services.segmentation_2d_service import segmentation_2d_service
from app.utils.serialization import base64_data_url_to_bytes

router = APIRouter(prefix="/classification", tags=["2D MRI Classification"])

ALLOWED_2D_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_IMAGE_SIZE_BYTES = 15 * 1024 * 1024  # 15 MB


@router.get("/models", summary="List available 2D classification models and checkpoints")
async def list_classification_models():
    """Lists registered classification models and their active checkpoint status."""
    return get_all_classification_models()


@router.post("/analyze", summary="Run 2D Brain MRI tumor classification with Grad-CAM")
async def analyze_2d_mri(
    image: UploadFile = File(..., description="2D Brain MRI scan in PNG or JPEG format"),
    model_id: Optional[str] = Form(None, description="Selected classification model identifier"),
    use_clahe: Optional[bool] = Form(None, description="Optional Contrast Limited Adaptive Histogram Equalization"),
    explain_segmentation: bool = Form(True, description="Whether to generate Seg-Grad-CAM explanation for 2D tumor segmentation"),
    explain_layer: Optional[str] = Form(None, description="Target layer for 2D Seg-Grad-CAM (e.g. conv2d_20 or up_sampling2d_2)"),
    current_user: UserDB = Depends(get_current_user_or_default),
):
    """
    Receives an individual 2D brain MRI scan.
    Validates file type (PNG, JPG, JPEG only) and file size.
    Validates model checkpoint availability.
    Applies model-specific preprocessing and runs inference.
    Generates class-activation Grad-CAM explanation.
    """
    filename = image.filename or "mri_scan.png"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext in {"nii", "gz"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="NIfTI volumetric files are not supported in 2D classification. Please switch to the 3D Segmentation workflow for .nii/.nii.gz scans.",
        )

    if ext not in ALLOWED_2D_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. 2D classification accepts only PNG, JPG, and JPEG files.",
        )

    content = await image.read(MAX_IMAGE_SIZE_BYTES + 1)
    if len(content) > MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of {MAX_IMAGE_SIZE_BYTES // (1024 * 1024)} MiB.",
        )

    effective_model_id = model_id or settings.DEFAULT_CLASSIFIER

    # Validate that model checkpoint exists
    if not model_loader.has_classification_model_for(effective_model_id):
        status_info = model_loader.get_status().get("classification_models", {}).get(effective_model_id, {})
        err_detail = status_info.get("reason") or status_info.get("error") or (
            f"Classification model '{effective_model_id}' trained checkpoint is not installed on the server. "
            f"Available models with checkpoints: EfficientNetV2B0."
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=err_detail)

    start_time = time.perf_counter()
    prediction_id = str(uuid.uuid4())

    try:
        # 1. Run Classification inference
        cls_result = await classification_service.predict(
            image_bytes=content,
            model_id=effective_model_id,
            use_clahe=use_clahe,
        )

        # 2. Run Grad-CAM explanation (model-aware layer and methods)
        gradcam_overlay = None
        gradcam_pp_overlay = None
        explanations_list = []

        target_layer = (
            "top_activation" if effective_model_id == "efficientnetv2b0_256"
            else ("swin_transformer_block_7" if effective_model_id == "swin_transformer" else "top_conv")
        )
        try:
            xai_result = await explainability_service.generate_explanation(
                image_bytes=content,
                analysis_id=prediction_id,
                method="gradcam",
                model_id=effective_model_id,
                target_layer=target_layer,
                target_class=cls_result.get("predicted_class"),
            )
            gradcam_overlay = xai_result.get("overlay_url")
            if xai_result.get("result_item"):
                item = xai_result["result_item"]
                explanations_list.append(item.model_dump() if hasattr(item, "model_dump") else dict(item))
        except Exception as xai_err:
            logger.warning(f"Grad-CAM generation failed: {xai_err}")

        # If model is EfficientNetV2B0, precompute Grad-CAM++ so it is immediately available
        if effective_model_id in ("efficientnetv2b0", "efficientnetv2b0_256"):
            try:
                pp_layer = "top_activation" if effective_model_id == "efficientnetv2b0_256" else "top_conv"
                xai_pp_result = await explainability_service.generate_explanation(
                    image_bytes=content,
                    analysis_id=prediction_id,
                    method="gradcam_pp",
                    model_id=effective_model_id,
                    target_layer=pp_layer,
                    target_class=cls_result.get("predicted_class"),
                )
                gradcam_pp_overlay = xai_pp_result.get("overlay_url")
                if xai_pp_result.get("result_item"):
                    item_pp = xai_pp_result["result_item"]
                    explanations_list.append(item_pp.model_dump() if hasattr(item_pp, "model_dump") else dict(item_pp))
            except Exception as pp_err:
                logger.warning(f"Grad-CAM++ precomputation failed: {pp_err}")

        # 3. Run Binary Tumor Segmentation — produce actual predicted tumor mask overlay
        segmentation_mask_url = None
        segmentation_overlay_url = None
        segmentation_model_used = None
        segmentation_tumor_detected = None
        segmentation_tumor_area_pct = None
        explain_segmentation_url = None
        explain_segmentation_layer = None
        explain_segmentation_feature_map = None
        explain_segmentation_caption = None
        explain_segmentation_heatmap_png = None
        explain_segmentation_stats = None
        explain_segmentation_available = False
        explain_obj = None
        try:
            seg_result = await segmentation_2d_service.segment_2d(
                image_bytes=content,
                analysis_id=prediction_id,
                explain=explain_segmentation,
                explain_layer=explain_layer,
            )
            segmentation_mask_url = seg_result.get("mask_url")
            segmentation_overlay_url = seg_result.get("overlay_url")
            segmentation_model_used = seg_result.get("model_used")
            segmentation_tumor_detected = seg_result.get("tumor_detected")
            segmentation_tumor_area_pct = seg_result.get("tumor_area_pct")
            explain_obj = seg_result.get("explain")
            explain_segmentation_url = seg_result.get("explain_url")
            explain_segmentation_layer = seg_result.get("explain_layer")
            explain_segmentation_feature_map = seg_result.get("explain_feature_map")
            explain_segmentation_caption = seg_result.get("explain_caption")
            explain_segmentation_heatmap_png = seg_result.get("explain_heatmap_png")
            explain_segmentation_stats = seg_result.get("explain_stats")
            explain_segmentation_available = seg_result.get("explain_available", False)
        except Exception as seg_err:
            logger.warning(f"2D tumor segmentation failed (non-fatal): {seg_err}")

        duration_ms = int((time.perf_counter() - start_time) * 1000)

        raw_scores = cls_result.get("scores_list") or []
        formatted_scores = [
            {
                "label": s.label,
                "score": round(float(s.score), 4),
                "percentage": round(float(s.score) * 100.0, 2),
                "isTopPrediction": s.isTopPrediction,
            }
            for s in raw_scores
        ]

        # Persist 2D classification analysis in MongoDB
        user_id = str(current_user.id) if (current_user and current_user.id and current_user.id != "guest_researcher") else "guest_researcher"
        raw_result = {
            "prediction_id": prediction_id,
            "top_prediction": cls_result.get("predicted_class"),
            "confidence": cls_result.get("confidence"),
            "scores": formatted_scores,
            "gradcam_overlay": gradcam_overlay,
            "gradcam_pp_overlay": gradcam_pp_overlay,
            "segmentation_mask_url": segmentation_mask_url,
            "segmentation_overlay_url": segmentation_overlay_url,
            "segmentation_model_used": segmentation_model_used,
            "segmentation_tumor_detected": segmentation_tumor_detected,
            "segmentation_tumor_area_pct": segmentation_tumor_area_pct,
            "explain": explain_obj,
            "explain_segmentation_url": explain_segmentation_url,
            "explain_segmentation_layer": explain_segmentation_layer,
            "explain_segmentation_feature_map": explain_segmentation_feature_map,
            "explain_segmentation_caption": explain_segmentation_caption,
            "explain_segmentation_heatmap_png": explain_segmentation_heatmap_png,
            "explain_segmentation_stats": explain_segmentation_stats,
            "explain_segmentation_available": explain_segmentation_available,
            "explanations": explanations_list,
            "processing_time_ms": duration_ms,
            "actual_model": cls_result.get("actual_model"),
            "model_id": effective_model_id,
            "model_name": cls_result.get("model_name", effective_model_id),
            "model_version": cls_result.get("model_version", "v2.5.0-clahe256"),
        }
        analysis_id = await analysis_service.save_classification_analysis(
            user_id=user_id,
            inference_result=raw_result,
            filename=filename,
            model_id=effective_model_id,
            model_name=cls_result.get("model_name", effective_model_id),
            model_version=cls_result.get("model_version", "v2.5.0-clahe256"),
            actual_model=cls_result.get("actual_model"),
            image_bytes=content,
        )

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "success": True,
                "analysis_id": analysis_id,
                "prediction_id": prediction_id,
                "workflow_type": "2D_CLASSIFICATION",
                "model_id": effective_model_id,
                "model_name": cls_result.get("model_name", effective_model_id),
                "model_version": cls_result.get("model_version", "v2.5.0-clahe256"),
                "actual_model": cls_result.get("actual_model"),
                "top_prediction": cls_result.get("predicted_class"),
                "confidence": cls_result.get("confidence"),
                "scores": formatted_scores,
                "gradcam_overlay": gradcam_overlay,
                "gradcam_pp_overlay": gradcam_pp_overlay,
                "segmentation_mask_url": segmentation_mask_url,
                "segmentation_overlay_url": segmentation_overlay_url,
                "segmentation_model_used": segmentation_model_used,
                "segmentation_tumor_detected": segmentation_tumor_detected,
                "segmentation_tumor_area_pct": segmentation_tumor_area_pct,
                "explain": explain_obj,
                "explain_segmentation_url": explain_segmentation_url,
                "explain_segmentation_heatmap_png": explain_segmentation_heatmap_png,
                "explain_segmentation_layer": explain_segmentation_layer,
                "explain_segmentation_feature_map": explain_segmentation_feature_map,
                "explain_segmentation_caption": explain_segmentation_caption,
                "explain_segmentation_stats": explain_segmentation_stats,
                "explain_segmentation_available": explain_segmentation_available,
                "processing_time_ms": duration_ms,
                "disclaimer": "Research model prediction only. Not a clinically validated diagnosis.",
            },
        )
    except ModelUnavailableException as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=exc.detail)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification inference error: {exc}",
        )


@router.post("/{analysis_id}/segment", summary="Generate 2D tumor segmentation mask for an existing analysis")
async def generate_2d_tumor_segmentation(
    analysis_id: str,
    explain: bool = True,
    explain_layer: Optional[str] = None,
    current_user: UserDB = Depends(get_current_user_or_default),
):
    """
    Runs the loaded 2D tumor segmentation model (DeepLabV3+ / BRISC Binary U-Net)
    on an existing 2D analysis image and persists the resulting mask & overlay URLs.
    """
    analysis = await analysis_service.get_analysis(
        analysis_id=analysis_id,
        user_id=current_user.id or "guest_researcher",
        is_admin=(current_user.role == "admin"),
    )
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis '{analysis_id}' was not found.",
        )

    image_source = getattr(analysis, "imageDataUrl", None) or getattr(analysis, "image_url", None)
    if not image_source:
        raise HTTPException(status_code=400, detail="Original MRI image not found for this analysis.")

    image_bytes = base64_data_url_to_bytes(image_source)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Could not load original MRI image bytes.")

    try:
        seg_result = await segmentation_2d_service.segment_2d(
            image_bytes=image_bytes,
            analysis_id=analysis_id,
            explain=explain,
            explain_layer=explain_layer,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"2D tumor segmentation failed: {exc}",
        )

    db = get_database()
    if db is not None:
        try:
            await db.analyses.update_one(
                {"analysis_id": analysis_id},
                {
                    "$set": {
                        "full_response.segmentation_mask_url": seg_result.get("mask_url"),
                        "full_response.segmentation_overlay_url": seg_result.get("overlay_url"),
                        "full_response.segmentation_model_used": seg_result.get("model_used"),
                        "full_response.segmentation_tumor_detected": seg_result.get("tumor_detected"),
                        "full_response.segmentation_tumor_area_pct": seg_result.get("tumor_area_pct"),
                        "full_response.explain": seg_result.get("explain"),
                        "full_response.explain_segmentation_url": seg_result.get("explain_url"),
                        "full_response.explain_segmentation_heatmap_png": seg_result.get("explain_heatmap_png"),
                        "full_response.explain_segmentation_layer": seg_result.get("explain_layer"),
                        "full_response.explain_segmentation_feature_map": seg_result.get("explain_feature_map"),
                        "full_response.explain_segmentation_caption": seg_result.get("explain_caption"),
                        "full_response.explain_segmentation_stats": seg_result.get("explain_stats"),
                        "full_response.explain_segmentation_available": seg_result.get("explain_available", True),
                    }
                },
            )
        except Exception as db_err:
            logger.warning("Failed to update MongoDB with 2D segmentation: %s", db_err)

    if analysis_id in _in_memory_analyses:
        fr = _in_memory_analyses[analysis_id].setdefault("full_response", {})
        fr["segmentation_mask_url"] = seg_result.get("mask_url")
        fr["segmentation_overlay_url"] = seg_result.get("overlay_url")
        fr["segmentation_model_used"] = seg_result.get("model_used")
        fr["segmentation_tumor_detected"] = seg_result.get("tumor_detected")
        fr["segmentation_tumor_area_pct"] = seg_result.get("tumor_area_pct")
        fr["explain"] = seg_result.get("explain")
        fr["explain_segmentation_url"] = seg_result.get("explain_url")
        fr["explain_segmentation_heatmap_png"] = seg_result.get("explain_heatmap_png")
        fr["explain_segmentation_layer"] = seg_result.get("explain_layer")
        fr["explain_segmentation_feature_map"] = seg_result.get("explain_feature_map")
        fr["explain_segmentation_caption"] = seg_result.get("explain_caption")
        fr["explain_segmentation_stats"] = seg_result.get("explain_stats")
        fr["explain_segmentation_available"] = seg_result.get("explain_available", True)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "success": True,
            "analysis_id": analysis_id,
            "segmentation_mask_url": seg_result.get("mask_url"),
            "segmentation_overlay_url": seg_result.get("overlay_url"),
            "segmentation_model_used": seg_result.get("model_used"),
            "segmentation_tumor_detected": seg_result.get("tumor_detected"),
            "segmentation_tumor_area_pct": seg_result.get("tumor_area_pct"),
            "explain": seg_result.get("explain"),
            "explain_segmentation_url": seg_result.get("explain_url"),
            "explain_segmentation_heatmap_png": seg_result.get("explain_heatmap_png"),
            "explain_segmentation_layer": seg_result.get("explain_layer"),
            "explain_segmentation_feature_map": seg_result.get("explain_feature_map"),
            "explain_segmentation_caption": seg_result.get("explain_caption"),
            "explain_segmentation_stats": seg_result.get("explain_stats"),
            "explain_segmentation_available": seg_result.get("explain_available", True),
        },
    )
