import json
import math
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

from app.api.deps import get_current_user_or_default
from app.core.exceptions import InvalidConfigurationException, ModelUnavailableException
from app.ml.model_loader import model_loader
from app.models.user import UserDB
from app.services.brats_3d_service import brats_3d_segmentation_service
from app.services.analysis_service import analysis_service

router = APIRouter(prefix="/segmentation", tags=["Volumetric Segmentation"])
MAX_NIFTI_FILE_BYTES = 128 * 1024 * 1024


def _json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


@router.get("/status", summary="Get BraTS 3D U-Net model status and system readiness")
async def get_segmentation_status():
    """Returns segmentation model configuration, weights readiness, GPU status, and architecture specs."""
    return brats_3d_segmentation_service.get_status()


@router.post("/validate-study", summary="Validate MRI study folder modalities, headers, and spatial alignment")
async def validate_mri_study(
    flair: Optional[UploadFile] = File(default=None, description="FLAIR modality NIfTI volume (.nii or .nii.gz)"),
    t1: Optional[UploadFile] = File(default=None, description="T1 modality NIfTI volume (.nii or .nii.gz)"),
    t1ce: Optional[UploadFile] = File(default=None, description="T1CE modality NIfTI volume (.nii or .nii.gz)"),
    t2: Optional[UploadFile] = File(default=None, description="T2 modality NIfTI volume (.nii or .nii.gz)"),
    seg: Optional[UploadFile] = File(default=None, description="Optional ground-truth segmentation volume (.nii or .nii.gz)"),
    current_user: UserDB = Depends(get_current_user_or_default),
):
    """
    Non-destructive pre-inference validation of an uploaded MRI study folder.
    Validates NIfTI format, 3D volume dimensions, voxel spacing, and spatial affine matrix alignment.
    Returns a comprehensive validation report to determine if the study is ready for 3D segmentation.
    """
    uploads = (("flair", flair), ("t1", t1), ("t1ce", t1ce), ("t2", t2))
    files = []
    for modality, upload in uploads:
        if upload is None or not upload.filename:
            continue
        filename = upload.filename
        content = await upload.read(MAX_NIFTI_FILE_BYTES + 1)
        if len(content) > MAX_NIFTI_FILE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"{modality.upper()} file '{filename}' exceeds 128 MiB size limit.",
            )
        files.append((modality, content, filename))

    ground_truth_file = None
    if seg is not None and seg.filename:
        content = await seg.read(MAX_NIFTI_FILE_BYTES + 1)
        ground_truth_file = (content, seg.filename)

    report = await run_in_threadpool(
        brats_3d_segmentation_service.validate_study,
        files,
        ground_truth_file,
    )
    return JSONResponse(content=_json_safe(report), status_code=status.HTTP_200_OK)


async def _execute_3d_segmentation_pipeline(
    flair: UploadFile,
    t1: UploadFile,
    t1ce: UploadFile,
    t2: UploadFile,
    seg: Optional[UploadFile],
    model_id: Optional[str],
    current_user: UserDB,
) -> Dict[str, Any]:
    valid_3d_models = {"brats_3d_unet", "unet3d_brats", "attention_cnn_vit", "attention_vit", "attention_vit_segmentation"}
    if model_id and model_id.lower() not in valid_3d_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model '{model_id}' is not a supported 3D multimodal segmentation model. Available 3D models: ['brats_3d_unet'].",
        )
    uploads = (("flair", flair), ("t1", t1), ("t1ce", t1ce), ("t2", t2))
    files = []
    for modality, upload in uploads:
        filename = upload.filename or f"{modality}.nii"
        if not (filename.lower().endswith(".nii") or filename.lower().endswith(".nii.gz")):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"{modality.upper()} file '{filename}' must be a NIfTI format (.nii or .nii.gz). Standard 2D images (PNG/JPG) are not supported.",
            )
        content = await upload.read(MAX_NIFTI_FILE_BYTES + 1)
        if len(content) > MAX_NIFTI_FILE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"{modality.upper()} file exceeds the 128 MiB size limit.",
            )
        files.append((modality, content, filename))

    ground_truth_file = None
    if seg is not None:
        filename = seg.filename or "seg.nii"
        if not (filename.lower().endswith(".nii") or filename.lower().endswith(".nii.gz")):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Ground-truth file '{filename}' must be a NIfTI format (.nii or .nii.gz).",
            )
        content = await seg.read(MAX_NIFTI_FILE_BYTES + 1)
        if len(content) > MAX_NIFTI_FILE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Ground-truth segmentation file exceeds the 128 MiB size limit.",
            )
        ground_truth_file = (content, filename)

    try:
        from app.services.storage_service import storage_service
        stored_modality_urls = {}
        for mod_name, file_bytes, fn in files:
            try:
                stored = storage_service.store_file("volumes_raw", file_bytes, fn)
                stored_modality_urls[mod_name] = stored["url"]
            except Exception:
                pass

        if ground_truth_file is not None:
            gt_bytes, gt_fn = ground_truth_file
            try:
                stored_gt = storage_service.store_file("volumes_raw", gt_bytes, gt_fn)
                stored_modality_urls["seg"] = stored_gt["url"]
            except Exception:
                pass

        target_model_id = model_id or "brats_3d_unet"
        result = await run_in_threadpool(
            brats_3d_segmentation_service.run_inference,
            files,
            ground_truth_file,
            True,
            target_model_id,
        )

        modality_filenames = {
            "flair": flair.filename or "flair.nii",
            "t1": t1.filename or "t1.nii",
            "t1ce": t1ce.filename or "t1ce.nii",
            "t2": t2.filename or "t2.nii",
        }
        if seg is not None:
            modality_filenames["seg"] = seg.filename or "seg.nii"

        loaded_info = model_loader._segmentation_model_info.get("brats_3d_unet", {})
        arch_title = loaded_info.get("architecture", "Attention CNN + Vision Transformer")
        user_id = str(current_user.id) if (current_user and current_user.id and current_user.id != "guest_researcher") else "guest_researcher"

        actual_model = result.get("actual_model") or {}
        model_name = result.get("model_name") or arch_title
        model_version = result.get("model_version") or loaded_info.get("version", "v2.4.0")

        analysis_id = await analysis_service.save_segmentation_analysis(
            user_id=user_id,
            inference_result=result,
            modality_filenames=modality_filenames,
            model_id=target_model_id,
            model_name=model_name,
            model_version=model_version,
            actual_model=actual_model,
        )

        input_shape = loaded_info.get("input_shape") or [96, 96, 64, 4]
        output_shape = loaded_info.get("output_shape") or [96, 96, 64, 4]

        class_mapping = {
            "0": "Background",
            "1": "Necrotic / non-enhancing tumor",
            "2": "Peritumoral edema",
            "3": "Enhancing tumor",
        }

        executed_model = {
            "model_id": target_model_id,
            "model_name": model_name,
            "model_version": model_version,
            "architecture": arch_title,
            "input_shape": input_shape,
            "output_shape": output_shape,
            "modalities": ["FLAIR", "T1", "T1CE", "T2"],
            "classes": ["Background", "Necrotic / non-enhancing tumor", "Peritumoral edema", "Enhancing tumor"],
            "parameters_count": actual_model.get("parameters_count", "3.07M params"),
            "framework": actual_model.get("framework", "TensorFlow 2.18 / Keras 3"),
        }

        response_payload = {
            **result,
            "analysis_id": analysis_id,
            "workflow_type": "3D_SEGMENTATION",
            "analysis_mode": "segmentation_3d",
            "processing_status": "completed",
            "executed_model": executed_model,
            "model_id": target_model_id,
            "model_name": model_name,
            "model_version": model_version,
            "input_modalities": ["FLAIR", "T1", "T1CE", "T2"],
            "original_volume_references": stored_modality_urls,
            "segmentation_mask_reference": result.get("download_url"),
            "segmentation_class_mapping": class_mapping,
            "inference_time": result.get("processing_time_ms", 180),
            "affine_metadata": {
                "orientation": result.get("anatomical_orientation", ["R", "A", "S"]),
                "dimensions": result.get("volume_dimensions"),
                "voxel_spacing": result.get("voxel_spacing"),
            },
        }

        return response_payload
    except (ModelUnavailableException, InvalidConfigurationException):
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"3D segmentation inference error: {exc}",
        )


@router.post("/analyze", summary="Run full BraTS 3D U-Net segmentation, metrics, and explainability")
async def analyze_segmentation_volume(
    flair: UploadFile = File(..., description="FLAIR modality NIfTI volume (.nii or .nii.gz)"),
    t1: UploadFile = File(..., description="T1 modality NIfTI volume (.nii or .nii.gz)"),
    t1ce: UploadFile = File(..., description="T1CE modality NIfTI volume (.nii or .nii.gz)"),
    t2: UploadFile = File(..., description="T2 modality NIfTI volume (.nii or .nii.gz)"),
    seg: Optional[UploadFile] = File(default=None, description="Optional ground-truth segmentation volume (.nii or .nii.gz)"),
    model_id: Optional[str] = Form(default="brats_3d_unet"),
    current_user: UserDB = Depends(get_current_user_or_default),
):
    """
    Receives four core MRI modalities (FLAIR, T1, T1CE, T2) + optional ground-truth segmentation.
    Executes sliding-window inference, computes evaluation metrics if SEG is provided,
    generates 3D Grad-CAM, bottleneck features, modality ablation, and multi-planar slice previews.
    Returns structured JSON with preview assets and a download URL for the generated NIfTI volume.
    """
    payload = await _execute_3d_segmentation_pipeline(flair, t1, t1ce, t2, seg, model_id, current_user)
    return JSONResponse(content=_json_safe(payload), status_code=status.HTTP_200_OK)


@router.post("/predict", summary="Inference endpoint for 3D multimodal brain tumor segmentation")
async def predict_segmentation_volume(
    flair: UploadFile = File(..., description="FLAIR modality NIfTI volume (.nii or .nii.gz)"),
    t1: UploadFile = File(..., description="T1 modality NIfTI volume (.nii or .nii.gz)"),
    t1ce: UploadFile = File(..., description="T1CE modality NIfTI volume (.nii or .nii.gz)"),
    t2: UploadFile = File(..., description="T2 modality NIfTI volume (.nii or .nii.gz)"),
    seg: Optional[UploadFile] = File(default=None, description="Optional ground-truth segmentation volume (.nii or .nii.gz)"),
    model_id: Optional[str] = Form(default="brats_3d_unet"),
    current_user: UserDB = Depends(get_current_user_or_default),
):
    """
    Standard contract endpoint for 3D multimodal brain tumor segmentation.
    Returns analysis_id, executed_model, class mappings, voxel dimensions, and spatial metadata.
    """
    payload = await _execute_3d_segmentation_pipeline(flair, t1, t1ce, t2, seg, model_id, current_user)
    return JSONResponse(content=_json_safe(payload), status_code=status.HTTP_200_OK)


@router.get("/download/{prediction_id}", summary="Download predicted segmentation NIfTI volume (.nii.gz)")
async def download_segmentation_nifti(prediction_id: str):
    """Streams the generated 3D segmentation NIfTI volume."""
    record = brats_3d_segmentation_service.get_prediction_bytes(prediction_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Segmentation prediction '{prediction_id}' not found or session has expired.",
        )
    nifti_bytes, filename = record
    return Response(
        content=nifti_bytes,
        media_type="application/gzip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


@router.get("/slice/{prediction_id}", summary="Get on-demand 2D cross-section slice PNG")
async def get_segmentation_slice(
    prediction_id: str,
    plane: str = "axial",
    index: int = 80,
    opacity: float = 0.65,
    modality: str = "flair",
    view_mode: str = "overlay",
    classes: Optional[str] = None,
):
    """
    Renders and streams an individual 2D cross-sectional slice PNG
    from the 3D volume for fluid multi-planar slice scrubbing.
    Supports view_mode ('overlay', 'mask_only', 'original') and classes ('1,2,3').
    """
    slice_png = brats_3d_segmentation_service.render_slice_at(
        prediction_id=prediction_id,
        plane=plane,
        index=index,
        opacity=opacity,
        modality=modality,
        view_mode=view_mode,
        classes=classes,
    )
    if slice_png is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Slice could not be rendered for prediction '{prediction_id}'."
        )
    return Response(content=slice_png, media_type="image/png", headers={"Cache-Control": "public, max-age=3600"})


@router.post("/brats-3d-unet", summary="Run BraTS 3D U-Net and download NIfTI directly")
async def segment_brats_volume_direct(
    flair: UploadFile = File(...),
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...),
    segmentation: UploadFile | None = File(default=None),
    current_user: UserDB = Depends(get_current_user_or_default),
):
    """Direct stream endpoint preserved for backward compatibility."""
    uploads = (("flair", flair), ("t1", t1), ("t1ce", t1ce), ("t2", t2))
    files = []
    for modality, upload in uploads:
        filename = upload.filename or f"{modality}.nii"
        content = await upload.read(MAX_NIFTI_FILE_BYTES + 1)
        if len(content) > MAX_NIFTI_FILE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"{modality.upper()} file exceeds 128 MiB limit.",
            )
        files.append((modality, content, filename))

    ground_truth_file = None
    if segmentation is not None:
        filename = segmentation.filename or "segmentation.nii"
        content = await segmentation.read(MAX_NIFTI_FILE_BYTES + 1)
        ground_truth_file = (content, filename)

    if not model_loader.has_segmentation_model_for("brats_3d_unet"):
        details = model_loader.get_status()["segmentation_models"].get("brats_3d_unet", {}).get("details", {})
        raise ModelUnavailableException(
            "brats_3d_unet",
            details.get("error") or details.get("reason") or "Checkpoint is not loaded and validated.",
        )

    try:
        result = await run_in_threadpool(
            brats_3d_segmentation_service.run_inference,
            files,
            ground_truth_file,
            False,
        )
        prediction_id = result["prediction_id"]
        cached = brats_3d_segmentation_service.get_prediction_bytes(prediction_id)
        if not cached:
            raise HTTPException(status_code=500, detail="Generated volume could not be retrieved.")
        nifti_bytes, filename = cached
        return Response(
            content=nifti_bytes,
            media_type="application/gzip",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Segmentation-Metadata": json.dumps(_json_safe(result), separators=(",", ":"), allow_nan=False),
            },
        )
    except (ModelUnavailableException, InvalidConfigurationException):
        raise
