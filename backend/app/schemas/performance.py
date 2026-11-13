from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class PerformanceModelMetadata(BaseModel):
    name: str
    version: str
    architecture: str
    framework: str
    inputSize: str
    totalParameters: str
    trainingDataset: str
    evaluationSplit: str
    lastUpdated: str
    evaluationProtocol: Optional[str] = None


class PerClassMetrics(BaseModel):
    className: str
    precision: float
    recall: float
    f1Score: float
    support: int


class ConfusionMatrixData(BaseModel):
    labels: List[str]
    matrix: List[List[int]]


class ModelMetricsResponse(BaseModel):
    accuracy: float
    macroPrecision: float
    macroRecall: float
    macroF1: float
    perClassMetrics: List[PerClassMetrics]
    confusionMatrix: ConfusionMatrixData
    metadata: PerformanceModelMetadata
    isExampleData: bool = False


# ─── Detailed Classification Model Item ────────────────────────────────────
class ClassificationModelItem(BaseModel):
    model_id: str
    name: str
    version: str
    architecture: str
    framework: str
    input_size: str
    parameters: str
    dataset: str
    evaluation_split: str
    last_evaluated: Optional[str] = None
    evaluated: bool = True
    status: str = "Evaluated"
    reason: Optional[str] = None
    accuracy: Optional[float] = None
    macro_precision: Optional[float] = None
    macro_recall: Optional[float] = None
    macro_f1: Optional[float] = None
    specificity: Optional[float] = None
    sensitivity: Optional[float] = None
    average_inference_time_ms: Optional[int] = None
    per_class_metrics: Optional[List[PerClassMetrics]] = None
    confusion_matrix: Optional[ConfusionMatrixData] = None


class ClassificationPerformanceResponse(BaseModel):
    models: List[ClassificationModelItem]
    best_model_id: str
    benchmark_dataset: str
    total_test_samples: int


# ─── Detailed Segmentation Model Item ──────────────────────────────────────
class PerRegionMetrics(BaseModel):
    region_name: str
    dice_score: float
    iou_score: float


class SegmentationModelItem(BaseModel):
    model_id: str
    name: str
    version: str
    architecture: str
    framework: str
    input_size: str
    parameters: str
    dataset: str
    evaluation_protocol: str
    last_evaluated: Optional[str] = None
    evaluated: bool = True
    status: str = "Evaluated"
    reason: Optional[str] = None
    dice_coefficient: Optional[float] = None  # Whole Tumor Dice
    mean_iou: Optional[float] = None         # Whole Tumor IoU
    precision: Optional[float] = None
    recall: Optional[float] = None
    sensitivity: Optional[float] = None
    specificity: Optional[float] = None
    pixel_accuracy: Optional[float] = None
    average_inference_time_ms: Optional[int] = None
    per_region_metrics: Optional[List[PerRegionMetrics]] = None
    classes: List[str] = Field(default_factory=list)


class SegmentationPerformanceResponse(BaseModel):
    models: List[SegmentationModelItem]
    best_model_id: str
    evaluation_cohort: str
    total_test_patients: int


class SegmentationMetricsResponse(BaseModel):
    model_name: str
    version: str
    dice_coefficient: float
    iou_score: float
    sensitivity: float
    evaluation_dataset: str
    classes: List[str]
    is_verified: bool = True


# ─── Unified Model Overview & Comparison ──────────────────────────────────
class UnifiedModelOverview(BaseModel):
    model_id: str
    name: str
    category: str  # "classification" | "segmentation"
    architecture: str
    parameters: str
    input_size: str
    evaluated: bool
    status: str
    primary_metric_label: str
    primary_metric_value: Optional[float] = None
    last_evaluated: Optional[str] = None


class ModelComparisonMatrix(BaseModel):
    classification_comparison: List[Dict[str, Any]]
    segmentation_comparison: List[Dict[str, Any]]
