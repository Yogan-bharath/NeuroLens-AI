from typing import List, Dict, Any
from fastapi import APIRouter
from app.services.metrics_service import metrics_service
from app.schemas.performance import (
    ModelMetricsResponse,
    ClassificationPerformanceResponse,
    SegmentationPerformanceResponse,
    UnifiedModelOverview,
    ModelComparisonMatrix,
)
from app.db.mongodb import get_database

router = APIRouter(prefix="/performance", tags=["Model Performance"])


@router.get("/summary", response_model=ModelMetricsResponse, summary="Get summary metrics for primary model")
async def get_performance_summary():
    """Retrieve verified evaluation metrics, per-class stats, and confusion matrix."""
    return metrics_service.get_summary_metrics()


@router.get("/models", response_model=List[UnifiedModelOverview], summary="List all models with evaluation status")
async def get_all_models_overview():
    """Returns overview of all classification and segmentation models, including evaluated status."""
    return metrics_service.get_all_models_overview()


@router.get("/classification", response_model=ClassificationPerformanceResponse, summary="Get classification benchmark metrics")
async def get_classification_performance():
    """
    Retrieve benchmark metrics for all classification models (EfficientNetV2B0, DenseNet121, ResNet50).
    Non-evaluated models explicitly return evaluated=False with 'Not evaluated' status.
    """
    return metrics_service.get_classification_metrics()


@router.get("/segmentation", response_model=SegmentationPerformanceResponse, summary="Get segmentation benchmark metrics")
async def get_segmentation_performance():
    """
    Retrieve verified benchmark metrics for segmentation models:
    - Attention CNN + Vision Transformer (89.37% WT Dice, 80.78% WT IoU)
    - Custom 3D U-Net (82.98% WT Dice, 72.83% WT IoU)
    - Attention U-Net (Not evaluated)
    - U-Net++ (Not evaluated)
    """
    return metrics_service.get_segmentation_metrics()


@router.get("/compare", response_model=ModelComparisonMatrix, summary="Compare models side-by-side")
async def compare_models():
    """Retrieve structured comparative metrics matrix across classification and segmentation models."""
    return metrics_service.get_comparison_data()


@router.get("/evaluation-history", summary="Retrieve historical model evaluation benchmark runs")
async def get_evaluation_history():
    """
    Retrieve historical offline benchmark runs across evaluation cohorts from MongoDB / verified registry.
    """
    db = get_database()
    if db is not None:
        try:
            cursor = db.model_evaluations.find({}, {"_id": 0}).sort("date", -1)
            records = await cursor.to_list(length=100)
            if records:
                return records
        except Exception:
            pass

    # Baseline historical evaluation records from official project benchmarks
    default_history = [
        {
            "run_id": "eval-brats2020-attvit-01",
            "model": "Attention CNN + Vision Transformer",
            "category": "segmentation",
            "dataset": "BraTS 2020 Multimodal MRI (56 Held-out Patients)",
            "evaluation_protocol": "Selected tumor-enriched test slices",
            "whole_tumor_dice": 0.8937,
            "whole_tumor_iou": 0.8078,
            "enhancing_tumor_dice": 0.8123,
            "tumor_core_dice": 0.7716,
            "parameters": "7.8M",
            "date": "2024-11-05",
            "verified": True,
            "status": "Evaluated"
        },
        {
            "run_id": "eval-brats2020-3dunet-02",
            "model": "Custom 3D U-Net",
            "category": "segmentation",
            "dataset": "BraTS 2020 Multimodal MRI (56 Held-out Patients)",
            "evaluation_protocol": "Complete 3D test volumes",
            "whole_tumor_dice": 0.8298,
            "whole_tumor_iou": 0.7283,
            "tumor_core_dice": 0.6961,
            "enhancing_tumor_dice": 0.6411,
            "parameters": "5.61M",
            "date": "2024-11-05",
            "verified": True,
            "status": "Evaluated"
        },
        {
            "run_id": "eval-effnetv2b0-clahe-03",
            "model": "EfficientNetV2B0",
            "category": "classification",
            "dataset": "BraTS 2021 + Figshare Brain Tumor MRI",
            "evaluation_protocol": "Independent 20% Held-out Test Split (N=1,311)",
            "accuracy": 0.941,
            "macro_f1": 0.936,
            "macro_precision": 0.938,
            "parameters": "7.1M",
            "date": "2024-10-18",
            "verified": True,
            "status": "Evaluated"
        },
        {
            "run_id": "eval-densenet121-pending",
            "model": "DenseNet121",
            "category": "classification",
            "dataset": "Figshare Brain MRI Benchmark",
            "evaluation_protocol": "Held-out Test Cohort",
            "accuracy": None,
            "macro_f1": None,
            "parameters": "8.0M",
            "date": "Pending",
            "verified": False,
            "status": "Not evaluated"
        },
        {
            "run_id": "eval-resnet50-pending",
            "model": "ResNet50",
            "category": "classification",
            "dataset": "Standard Brain Tumor MRI Test Split",
            "evaluation_protocol": "Held-out Test Cohort",
            "accuracy": None,
            "macro_f1": None,
            "parameters": "25.6M",
            "date": "Pending",
            "verified": False,
            "status": "Not evaluated"
        },
        {
            "run_id": "eval-attention-unet-pending",
            "model": "Attention U-Net",
            "category": "segmentation",
            "dataset": "BraTS 2020 Benchmark",
            "evaluation_protocol": "Complete 3D test volumes",
            "whole_tumor_dice": None,
            "whole_tumor_iou": None,
            "parameters": "6.4M",
            "date": "Pending",
            "verified": False,
            "status": "Not evaluated"
        },
        {
            "run_id": "eval-unet-plus-plus-pending",
            "model": "U-Net++",
            "category": "segmentation",
            "dataset": "BraTS 2020 Benchmark",
            "evaluation_protocol": "Complete 3D test volumes",
            "whole_tumor_dice": None,
            "whole_tumor_iou": None,
            "parameters": "9.1M",
            "date": "Pending",
            "verified": False,
            "status": "Not evaluated"
        }
    ]

    # Seed MongoDB if empty
    if db is not None:
        try:
            count = await db.model_evaluations.count_documents({})
            if count == 0:
                await db.model_evaluations.insert_many(default_history)
        except Exception:
            pass

    return default_history
