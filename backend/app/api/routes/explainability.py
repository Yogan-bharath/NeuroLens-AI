from typing import List
from fastapi import APIRouter, Depends, HTTPException, status

from app.models.user import UserDB
from app.api.deps import get_current_user_or_default
from app.services.analysis_service import analysis_service, _in_memory_analyses
from app.services.explainability_service import explainability_service
from app.utils.serialization import base64_data_url_to_bytes
from app.schemas.explainability import GenerateExplanationRequest, ExplanationArtifactResponse
from app.schemas.analysis import ExplanationResultItem
from app.db.mongodb import get_database
from app.core.logging import logger

router = APIRouter(prefix="/analyses", tags=["Explainability"])


@router.post("/{analysis_id}/explain", response_model=ExplanationArtifactResponse)
async def generate_analysis_explanation(
    analysis_id: str,
    request: GenerateExplanationRequest,
    current_user: UserDB = Depends(get_current_user_or_default)
):
    """
    Generate on-demand explanation (e.g. Grad-CAM, Grad-CAM++, Integrated Gradients)
    targeting a specified convolutional feature layer or class.
    """
    analysis = await analysis_service.get_analysis(
        analysis_id=analysis_id,
        user_id=current_user.id or "guest_researcher",
        is_admin=(current_user.role == "admin")
    )
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis session '{analysis_id}' was not found."
        )

    # Decode image bytes from input_image url or base64 data url
    if not analysis.imageDataUrl:
        raise HTTPException(status_code=400, detail="Original image data not found for analysis.")

    image_bytes = base64_data_url_to_bytes(analysis.imageDataUrl)

    target_class = request.target_class or analysis.predictedClass or "Glioma"
    actual_m = analysis.actual_model
    model_id = (
        (actual_m.model_id if actual_m and actual_m.model_id else None) or
        analysis.model_id or
        (analysis.metadata.modelName.lower().replace(" ", "_") if analysis.metadata else "efficientnetv2b0")
    )

    # Model-aware layer resolution
    target_layer = request.target_layer
    if not target_layer or target_layer == "top_conv":
        target_layer = "swin_transformer_block_7" if model_id == "swin_transformer" else "top_conv"

    result = await explainability_service.generate_explanation(
        image_bytes=image_bytes,
        analysis_id=analysis_id,
        method=request.method,
        model_id=model_id,
        target_layer=target_layer,
        target_class=target_class,
        target_region=request.target_region
    )

    # Persist the newly computed explanation to MongoDB and in-memory cache
    result_item = result.get("result_item")
    if result_item:
        item_dict = result_item.model_dump() if hasattr(result_item, "model_dump") else dict(result_item)
        overlay_url = result.get("overlay_url")
        db = get_database()
        if db is not None:
            try:
                update_fields: dict = {
                    "$push": {
                        "explainability": item_dict,
                        "full_response.explanations": item_dict
                    }
                }
                set_fields = {}
                if result["method"] == "gradcam_pp":
                    set_fields["gradcam_pp_overlay"] = overlay_url
                    set_fields["full_response.gradcam_pp_overlay"] = overlay_url
                elif result["method"] == "gradcam":
                    set_fields["gradcam_overlay"] = overlay_url
                    set_fields["full_response.gradcam_overlay"] = overlay_url
                if set_fields:
                    update_fields["$set"] = set_fields

                await db.analyses.update_one(
                    {"analysis_id": analysis_id},
                    update_fields
                )
            except Exception as e:
                logger.warning(f"Failed to persist new explanation to MongoDB: {e}")

        if analysis_id in _in_memory_analyses:
            doc = _in_memory_analyses[analysis_id]
            if "explainability" in doc and isinstance(doc["explainability"], list):
                doc["explainability"].append(item_dict)
            if "full_response" in doc and isinstance(doc["full_response"], dict):
                exps = doc["full_response"].setdefault("explanations", [])
                if isinstance(exps, list):
                    exps.append(item_dict)
                if result["method"] == "gradcam_pp":
                    doc["full_response"]["gradcam_pp_overlay"] = overlay_url
                    doc["gradcam_pp_overlay"] = overlay_url
                elif result["method"] == "gradcam":
                    doc["full_response"]["gradcam_overlay"] = overlay_url
                    doc["gradcam_overlay"] = overlay_url
                    exps.append(item_dict)

    return ExplanationArtifactResponse(
        analysis_id=analysis_id,
        method=result["method"],
        target_layer=result["target_layer"],
        target_class=result.get("target_class"),
        target_region=result.get("target_region"),
        heatmap_url=result["heatmap_url"],
        overlay_url=result["overlay_url"],
        description=result["description"],
        metadata={"is_mock": result.get("is_mock", False)}
    )


@router.get("/{analysis_id}/explanations", response_model=List[ExplanationResultItem])
async def get_analysis_explanations(
    analysis_id: str,
    current_user: UserDB = Depends(get_current_user_or_default)
):
    """Retrieve all generated explainability overlays for an analysis session."""
    analysis = await analysis_service.get_analysis(
        analysis_id=analysis_id,
        user_id=current_user.id or "guest_researcher",
        is_admin=(current_user.role == "admin")
    )
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis session '{analysis_id}' was not found."
        )
    return analysis.explanations
