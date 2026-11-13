from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, status

from app.ml.model_registry import (
    get_all_classification_models,
    get_all_segmentation_models,
    get_all_xai_methods,
    get_model_by_id,
)
from app.models.model_registry import (
    ClassificationModelInfo,
    SegmentationModelInfo,
    XaiMethodInfo,
)

router = APIRouter(prefix="/models", tags=["Models & Configuration"])


@router.get("", response_model=Dict[str, Any])
async def list_all_models():
    """Retrieve full catalog of classification, segmentation, and XAI models."""
    return {
        "classification": get_all_classification_models(),
        "segmentation": get_all_segmentation_models(),
        "xai_methods": get_all_xai_methods(),
    }


@router.get("/classification", response_model=List[ClassificationModelInfo])
async def list_classification_models():
    """Retrieve registered classification models and verified metrics."""
    return get_all_classification_models()


@router.get("/segmentation", response_model=List[SegmentationModelInfo])
async def list_segmentation_models():
    """Retrieve registered segmentation models and class definitions."""
    return get_all_segmentation_models()


@router.get("/xai-methods", response_model=List[XaiMethodInfo])
async def list_xai_methods():
    """Retrieve explainable AI methods and architectural layer compatibility."""
    return get_all_xai_methods()


@router.get("/compare", response_model=Dict[str, Any])
async def compare_models():
    """Retrieve cross-model benchmark comparison matrix."""
    return {
        "classification": [
            {
                "id": m.id,
                "name": m.name,
                "architecture": m.architecture,
                "accuracy": m.metrics.testAccuracy if m.metrics else None,
                "macro_f1": m.metrics.macroF1 if m.metrics else None,
                "inference_ms": m.metrics.inferenceTimeMs if m.metrics else None,
                "status": m.status
            }
            for m in get_all_classification_models()
        ],
        "segmentation": [
            {
                "id": m.id,
                "name": m.name,
                "architecture": m.architecture,
                "dice": m.metrics.diceCoefficient if m.metrics else None,
                "iou": m.metrics.iouScore if m.metrics else None,
                "status": m.status
            }
            for m in get_all_segmentation_models()
        ]
    }


@router.get("/{model_id}", response_model=Dict[str, Any])
async def get_model_details(model_id: str):
    """Retrieve architectural details, layer maps, and verified metrics for a model."""
    model = get_model_by_id(model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model with identifier '{model_id}' was not found in registry."
        )
    return model.model_dump()
