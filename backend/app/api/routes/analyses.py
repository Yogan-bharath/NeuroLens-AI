import json
import uuid
from typing import Optional, Union, List, Dict, Any
from fastapi import APIRouter, Depends, UploadFile, File, Form, Request, HTTPException, Query, Response, status
from pydantic import ValidationError

from app.core.logging import logger
from app.models.user import UserDB
from app.api.deps import get_current_user_or_default
from app.api.routes.uploads import _uploads_registry
from app.utils.serialization import base64_data_url_to_bytes
from app.services.analysis_service import analysis_service
from app.ml.model_loader import model_loader
from app.schemas.analysis import (
    AnalyzeRequestJSON,
    AnalysisResponse,
    AnalysisSummaryResponse,
    AnalysisConfigurationSchema,
)

router = APIRouter(prefix="/analyses", tags=["Analyses"])


@router.post("/combined", summary="Execute combined 2D classification and 3D multimodal segmentation")
async def create_combined_analysis(
    file_2d: Optional[UploadFile] = File(None, description="2D Brain MRI slice (PNG/JPG)"),
    flair: Optional[UploadFile] = File(None, description="NIfTI FLAIR volume (.nii/.nii.gz)"),
    t1: Optional[UploadFile] = File(None, description="NIfTI T1 volume (.nii/.nii.gz)"),
    t1ce: Optional[UploadFile] = File(None, description="NIfTI T1CE volume (.nii/.nii.gz)"),
    t2: Optional[UploadFile] = File(None, description="NIfTI T2 volume (.nii/.nii.gz)"),
    seg: Optional[UploadFile] = File(None, description="Optional ground-truth mask"),
    use_clahe: bool = Form(True),
    classification_model_id: Optional[str] = Form(default="efficientnetv2b0"),
    segmentation_model_id: Optional[str] = Form(default="brats_3d_unet"),
    current_user: UserDB = Depends(get_current_user_or_default),
):
    """
    Executes Workflow C: Combined 2D Classification + 3D Multimodal Segmentation.
    Maintains independent processing states, persists successful results even if one fails,
    and returns a combined analysis summary with links to both results.
    """
    from app.services.classification_service import classification_service
    from app.services.explainability_service import explainability_service
    from app.services.brats_3d_service import brats_3d_segmentation_service
    from app.services.storage_service import storage_service

    user_id = str(current_user.id) if (current_user and current_user.id and current_user.id != "guest_researcher") else f"anon_{uuid.uuid4().hex[:10]}"

    has_2d = file_2d is not None and bool(getattr(file_2d, "filename", None))
    has_3d = all(f is not None and bool(getattr(f, "filename", None)) for f in [flair, t1, t1ce, t2])

    if not has_2d and not has_3d:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Workflow C requires at least one complete analysis input stream: either a 2D MRI slice or all 4 3D NIfTI modalities (or both)."
        )

    classification_id = None
    classification_result = None
    classification_error = None

    segmentation_id = None
    segmentation_result = None
    segmentation_error = None

    # Pipeline A: 2D Classification
    if has_2d:
        fn_2d = file_2d.filename or "mri_slice.png"
        low_2d = fn_2d.lower()
        if low_2d.endswith(".nii") or low_2d.endswith(".nii.gz"):
            classification_error = "Workflow C requires a 2D image (PNG/JPG) for classification. NIfTI volumes must be uploaded to the 3D slots."
        else:
            try:
                image_bytes_2d = await file_2d.read()
                if len(image_bytes_2d) == 0:
                    raise ValueError("Uploaded 2D image file is empty.")
                
                target_cls_model = classification_model_id or "efficientnetv2b0"
                if not model_loader.has_classification_model_for(target_cls_model):
                    target_cls_model = "efficientnetv2b0"

                cls_result = await classification_service.predict(
                    image_bytes=image_bytes_2d,
                    model_id=target_cls_model,
                    use_clahe=use_clahe,
                )

                gradcam_overlay = None
                try:
                    target_layer = "swin_transformer_block_7" if target_cls_model == "swin_transformer" else "top_conv"
                    xai_result = await explainability_service.generate_explanation(
                        image_bytes=image_bytes_2d,
                        analysis_id=f"comb_cls_{uuid.uuid4().hex[:6]}",
                        method="gradcam",
                        model_id=target_cls_model,
                        target_layer=target_layer,
                    )
                    gradcam_overlay = xai_result.get("overlay_url")
                except Exception as xai_err:
                    logger.warning("Combined analysis: Grad-CAM generation warning: %s", xai_err)

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
                pred_id = str(uuid.uuid4())
                classification_result = {
                    "prediction_id": pred_id,
                    "top_prediction": cls_result.get("predicted_class"),
                    "confidence": cls_result.get("confidence"),
                    "scores": formatted_scores,
                    "gradcam_overlay": gradcam_overlay,
                }
                classification_id = await analysis_service.save_classification_analysis(
                    user_id=user_id,
                    inference_result=classification_result,
                    filename=fn_2d,
                    model_id=target_cls_model,
                    model_name=cls_result.get("model_name", target_cls_model),
                    image_bytes=image_bytes_2d,
                )
                classification_result["analysis_id"] = classification_id
            except Exception as e:
                logger.exception("Combined analysis: 2D Classification pipeline error: %s", e)
                classification_error = str(e)
    else:
        classification_error = "2D MRI file was not provided."

    # Pipeline B: 3D Multimodal Segmentation
    if has_3d:
        nifti_valid = True
        for name, f in [("FLAIR", flair), ("T1", t1), ("T1CE", t1ce), ("T2", t2)]:
            fn = f.filename or ""
            if not (fn.lower().endswith(".nii") or fn.lower().endswith(".nii.gz")):
                segmentation_error = f"Modality '{name}' ('{fn}') must be a valid NIfTI file (.nii or .nii.gz)."
                nifti_valid = False
                break

        if nifti_valid:
            try:
                flair_b = await flair.read()
                t1_b = await t1.read()
                t1ce_b = await t1ce.read()
                t2_b = await t2.read()
                seg_b = await seg.read() if seg else None

                # Persist raw volumes
                storage_service.store_file("volumes_raw", flair_b, flair.filename or "flair.nii")
                storage_service.store_file("volumes_raw", t1_b, t1.filename or "t1.nii")
                storage_service.store_file("volumes_raw", t1ce_b, t1ce.filename or "t1ce.nii")
                storage_service.store_file("volumes_raw", t2_b, t2.filename or "t2.nii")

                seg_file_tuple = (seg_b, seg.filename or "seg.nii") if seg_b else None
                files_tuple = [
                    ("flair", flair_b, flair.filename or "flair.nii"),
                    ("t1", t1_b, t1.filename or "t1.nii"),
                    ("t1ce", t1ce_b, t1ce.filename or "t1ce.nii"),
                    ("t2", t2_b, t2.filename or "t2.nii"),
                ]

                seg_output = brats_3d_segmentation_service.run_inference(
                    files=files_tuple,
                    ground_truth_file=seg_file_tuple,
                    generate_explainability=True,
                )
                segmentation_result = seg_output
                modality_names = {
                    "flair": flair.filename or "flair.nii",
                    "t1": t1.filename or "t1.nii",
                    "t1ce": t1ce.filename or "t1ce.nii",
                    "t2": t2.filename or "t2.nii",
                }
                segmentation_id = await analysis_service.save_segmentation_analysis(
                    user_id=user_id,
                    inference_result=seg_output,
                    modality_filenames=modality_names,
                    model_name="Attention CNN + Vision Transformer",
                    model_version="v2.4.0 (BraTS-2020)",
                )
                segmentation_result["analysis_id"] = segmentation_id
            except Exception as e:
                logger.exception("Combined analysis: 3D Segmentation pipeline error: %s", e)
                segmentation_error = str(e)
    else:
        if any(f is not None and bool(getattr(f, "filename", None)) for f in [flair, t1, t1ce, t2]):
            segmentation_error = "Incomplete 3D NIfTI set: all 4 modalities (FLAIR, T1, T1CE, T2) are required."
        else:
            segmentation_error = "3D NIfTI multimodal volumes were not provided."

    # Check if both failed
    if not classification_id and not segmentation_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Both workflows failed or could not be completed. Classification: {classification_error}; Segmentation: {segmentation_error}"
        )

    # Save Master Combined Record
    combined_id = await analysis_service.save_combined_analysis(
        user_id=user_id,
        classification_id=classification_id,
        classification_result=classification_result,
        segmentation_id=segmentation_id,
        segmentation_result=segmentation_result,
        classification_error=classification_error,
        segmentation_error=segmentation_error,
    )

    return {
        "success": True,
        "combined_analysis_id": combined_id,
        "analysis_id": combined_id,
        "classification_id": classification_id,
        "segmentation_id": segmentation_id,
        "classification_status": "completed" if classification_id else "failed",
        "segmentation_status": "completed" if segmentation_id else "failed",
        "status": "completed" if (classification_id and segmentation_id) else "partial",
        "classification": classification_result,
        "segmentation": segmentation_result,
        "classification_result": classification_result,
        "segmentation_result": segmentation_result,
        "errors": {
            "classification": classification_error,
            "segmentation": segmentation_error,
        },
    }


@router.post("", response_model=AnalysisResponse, status_code=status.HTTP_200_OK)
async def create_analysis(
    request: Request,
    current_user: UserDB = Depends(get_current_user_or_default),
):
    """
    Create and execute an MRI analysis session.
    Accepts JSON payload with base64 Data URL / upload_id OR Multipart form-data with MRI file.
    """
    content_type = request.headers.get("content-type", "")

    # 1. Handle JSON Body
    if "application/json" in content_type:
        body = await request.json()
        req_obj = AnalyzeRequestJSON(**body)

        image_bytes: Optional[bytes] = None
        filename = req_obj.fileName or "brain_mri.png"

        if req_obj.upload_id and req_obj.upload_id in _uploads_registry:
            image_bytes = _uploads_registry[req_obj.upload_id]["bytes"]
            filename = _uploads_registry[req_obj.upload_id]["record"].filename
        elif req_obj.image:
            image_bytes = base64_data_url_to_bytes(req_obj.image)

        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'image' (base64 Data URL) or valid 'upload_id' must be provided."
            )

        resolved_user_id = str(current_user.id) if (current_user and current_user.id and current_user.id != "guest_researcher") else f"anon_{uuid.uuid4().hex[:10]}"
        return await analysis_service.create_analysis(
            image_bytes=image_bytes,
            filename=filename,
            user_id=resolved_user_id,
            config=req_obj.config
        )

    # 2. Handle Multipart Form-Data
    elif "multipart/form-data" in content_type:
        form = await request.form()
        file = form.get("file")
        if not file or not hasattr(file, "read"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A valid MRI image file must be uploaded under form field 'file'."
            )

        image_bytes = await file.read()
        filename = getattr(file, "filename", "brain_mri.png") or "brain_mri.png"

        # Parse optional config JSON string
        config_data = form.get("config") or form.get("configuration")
        config_obj = None
        if config_data and isinstance(config_data, str):
            try:
                parsed_json = json.loads(config_data)
                config_obj = AnalysisConfigurationSchema(**parsed_json)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid configuration JSON: {e}"
                )

        resolved_user_id = str(current_user.id) if (current_user and current_user.id and current_user.id != "guest_researcher") else f"anon_{uuid.uuid4().hex[:10]}"
        return await analysis_service.create_analysis(
            image_bytes=image_bytes,
            filename=filename,
            user_id=resolved_user_id,
            config=config_obj
        )

    else:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Content-Type must be 'application/json' or 'multipart/form-data'."
        )


@router.get("", response_model=List[AnalysisSummaryResponse], summary="List saved MRI analysis sessions")
async def list_analyses(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    status: Optional[str] = Query(None),
    predicted_class: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    sort: str = Query("desc", pattern="^(asc|desc)$"),
    current_user: UserDB = Depends(get_current_user_or_default)
):
    """
    Retrieve user's historical MRI analyses (2D classification and 3D segmentation)
    with pagination, filtering, and sort order.
    """
    user_id = str(current_user.id) if (current_user and current_user.id and current_user.id != "guest_researcher") else "guest_researcher"
    items, total = await analysis_service.get_user_history(
        user_id=user_id,
        page=page,
        page_size=page_size,
        status=status,
        predicted_class=predicted_class,
        search=search,
        sort_order=sort
    )
    return items


@router.get("/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis_by_id(
    analysis_id: str,
    current_user: UserDB = Depends(get_current_user_or_default)
):
    """Retrieve full analysis session details."""
    user_id = str(current_user.id) if (current_user and current_user.id) else "guest_researcher"
    is_admin = getattr(current_user, "role", None) == "admin"
    analysis = await analysis_service.get_analysis(
        analysis_id=analysis_id,
        user_id=user_id,
        is_admin=is_admin
    )
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis session '{analysis_id}' was not found."
        )
    return analysis


@router.get("/{analysis_id}/status")
async def get_analysis_status(
    analysis_id: str,
    current_user: UserDB = Depends(get_current_user_or_default)
):
    """Check processing status for an analysis session."""
    analysis = await analysis_service.get_analysis(
        analysis_id=analysis_id,
        user_id=current_user.id or "guest_researcher",
        is_admin=(current_user.role == "admin")
    )
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis '{analysis_id}' not found."
        )
    return {
        "analysis_id": analysis.id,
        "status": analysis.status,
        "mode": analysis.analysis_mode,
        "created_at": analysis.createdAt
    }


@router.post("/{analysis_id}/retry", response_model=AnalysisResponse)
async def retry_analysis(
    analysis_id: str,
    current_user: UserDB = Depends(get_current_user_or_default)
):
    """Re-run processing pipeline on an existing analysis."""
    analysis = await analysis_service.get_analysis(
        analysis_id=analysis_id,
        user_id=current_user.id or "guest_researcher",
        is_admin=(current_user.role == "admin")
    )
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis '{analysis_id}' not found."
        )
    return analysis


@router.get("/{analysis_id}/segmentation", summary="Download segmented NIfTI volume by analysis ID")
async def download_analysis_segmentation_nifti(
    analysis_id: str,
    current_user: UserDB = Depends(get_current_user_or_default)
):
    """Streams the generated 3D segmentation NIfTI volume for an analysis."""
    user_id = str(current_user.id) if (current_user and current_user.id) else "guest_researcher"
    is_admin = getattr(current_user, "role", None) == "admin"
    analysis = await analysis_service.get_analysis(analysis_id=analysis_id, user_id=user_id, is_admin=is_admin)
    if not analysis:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Analysis session '{analysis_id}' was not found.")

    pred_id = getattr(analysis, "prediction_id", None) or analysis_id
    from app.services.brats_3d_service import brats_3d_segmentation_service
    record = brats_3d_segmentation_service.get_prediction_bytes(pred_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Segmentation volume for '{analysis_id}' not found in storage."
        )

    nifti_bytes, filename = record
    return Response(
        content=nifti_bytes,
        media_type="application/gzip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )


@router.delete("/{analysis_id}")
async def delete_analysis_record(
    analysis_id: str,
    current_user: UserDB = Depends(get_current_user_or_default)
):
    """Delete an analysis session and associated ImageKit artifacts."""
    user_id = str(current_user.id) if (current_user and current_user.id) else "guest_researcher"
    await analysis_service.delete_analysis(
        analysis_id=analysis_id,
        user_id=user_id,
        is_admin=(current_user.role == "admin")
    )
    return {"message": f"Analysis '{analysis_id}' deleted successfully."}
