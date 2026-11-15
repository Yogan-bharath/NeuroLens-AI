from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, model_validator, ConfigDict


# ─── Configuration Schemas ──────────────────────────────────────────────
class XAIConfig(BaseModel):
    method: str = "gradcam"
    target_layer: Optional[str] = "top_conv"
    target_region: Optional[str] = "tumor_core"


class VisualizationConfig(BaseModel):
    layout: Optional[str] = "side_by_side"
    overlay_opacity: float = 0.7


class AnalysisConfigurationSchema(BaseModel):
    mode: str = Field(default="combined", description="classification | segmentation | combined")
    classification_model_id: str = Field(default="efficientnetv2b0", alias="classificationModelId")
    segmentation_model_id: str = Field(default="unet", alias="segmentationModelId")
    classification_xai_method: str = Field(default="gradcam", alias="classificationXaiMethod")
    classification_target: str = Field(default="Predicted Top Class", alias="classificationTarget")
    classification_layer: str = Field(default="top_conv", alias="classificationLayer")
    segmentation_xai_method: str = Field(default="seg_gradcam", alias="segmentationXaiMethod")
    segmentation_layer: str = Field(default="bottleneck.conv2", alias="segmentationLayer")
    segmentation_visualization: str = Field(default="comparison", alias="segmentationVisualization")
    segmentation_opacity: float = Field(default=0.7, alias="segmentationOpacity")
    model_config = ConfigDict(populate_by_name=True)


# ─── Request Schemas ────────────────────────────────────────────────────
class AnalyzeRequestJSON(BaseModel):
    image: Optional[str] = Field(None, description="Base64 data URL or remote URL of MRI image")
    upload_id: Optional[str] = Field(None, description="Existing upload record ID")
    fileName: Optional[str] = Field("brain_mri.png", description="Original filename")
    config: Optional[AnalysisConfigurationSchema] = None


# ─── Component Result Schemas ───────────────────────────────────────────
class ActualModelMetadata(BaseModel):
    model_id: str
    model_name: str
    model_version: str
    architecture: str
    framework: str = "TensorFlow 2.18 / Keras 3"
    task_type: str  # "2D Brain Tumor Classification" | "3D Multimodal Volumetric Segmentation"
    input_resolution: Optional[str] = None
    input_modalities: Optional[List[str]] = None
    input_dimensions: Optional[List[int]] = None
    voxel_spacing: Optional[List[float]] = None
    preprocessing: Optional[str] = None
    parameters_count: Optional[str] = None
    inference_time_ms: Optional[int] = None
    inference_timestamp: Optional[str] = None
    is_mock: bool = False
    model_config = ConfigDict(extra="allow")


class ClassificationScore(BaseModel):
    label: str
    score: float
    isTopPrediction: bool


class ExplanationResultItem(BaseModel):
    method: str
    heatmapDataUrl: str
    overlayDataUrl: str
    targetClass: str
    layerName: str
    description: str
    featureMapShape: Optional[str] = None
    disclaimer: Optional[str] = None


class SegmentationClassItem(BaseModel):
    label: str
    color: str
    pixelCount: int
    percentage: float


class SegmentationResultItem(BaseModel):
    maskDataUrl: str
    overlayDataUrl: str
    classes: List[SegmentationClassItem]
    description: str
    dice_coefficient: Optional[float] = None
    iou: Optional[float] = None


class AnalysisMetadataItem(BaseModel):
    modelName: str
    modelVersion: str
    processingTimeMs: Optional[int] = 0
    imageSize: Optional[Dict[str, int]] = Field(default_factory=lambda: {"width": 256, "height": 256})
    analysisDate: str


class InputImageData(BaseModel):
    filename: str
    url: str
    width: int
    height: int
    file_id: Optional[str] = None


class DetailedClassification(BaseModel):
    model_id: str
    model_version: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    inference_time_ms: int


class DetailedSegmentation(BaseModel):
    model_id: str
    regions: List[SegmentationClassItem] = Field(default_factory=list)
    mask_url: Optional[str] = None
    overlay_url: Optional[str] = None
    dice_coefficient: Optional[float] = None
    iou: Optional[float] = None


class DetailedExplainability(BaseModel):
    classification: Optional[Dict[str, Any]] = None
    segmentation: Optional[Dict[str, Any]] = None


# ─── Main Analysis Response Schema ──────────────────────────────────────
class AnalysisResponse(BaseModel):
    # Primary IDs (supports frontend and section 8)
    id: str
    analysis_id: str
    fileName: str
    imageDataUrl: Optional[str] = None
    status: str = "completed"  # idle, uploading, preprocessing, classifying, explaining, segmenting, completed, failed
    analysis_mode: str = "classification_2d"  # classification_2d | segmentation_3d | combined
    workflow_type: Optional[str] = "2D_CLASSIFICATION"  # 2D_CLASSIFICATION | 3D_SEGMENTATION

    # Frontend-specific direct outputs
    classification: Optional[Union[List[ClassificationScore], Dict[str, Any]]] = None
    predictedClass: Optional[str] = None
    confidence: Optional[float] = None
    explanations: List[Union[ExplanationResultItem, Dict[str, Any]]] = Field(default_factory=list)
    segmentation: Optional[Union[SegmentationResultItem, Dict[str, Any]]] = None
    metadata: Optional[Union[AnalysisMetadataItem, Dict[str, Any]]] = None
    isDemo: bool = False
    createdAt: str

    # Section 8 detailed contracts
    input_image: Optional[InputImageData] = None
    classification_details: Optional[DetailedClassification] = None
    segmentation_details: Optional[DetailedSegmentation] = None
    explainability_details: Optional[DetailedExplainability] = None

    # Volumetric 3D & multimodal fields
    volume_dimensions: Optional[List[int]] = None
    voxel_spacing: Optional[List[float]] = None
    anatomical_orientation: Optional[List[str]] = None
    class_distribution: Optional[List[Dict[str, Any]]] = None
    slices: Optional[Dict[str, Any]] = None
    surface_3d: Optional[Dict[str, Any]] = None
    download_url: Optional[str] = None
    evaluation_metrics: Optional[Dict[str, Any]] = None
    modalities: Optional[List[str]] = None
    model_name: Optional[str] = None
    model_id: Optional[str] = None
    model_version: Optional[str] = None
    actual_model: Optional[ActualModelMetadata] = None
    model_info: Optional[Dict[str, Any]] = None
    top_prediction: Optional[str] = None
    scores: Optional[List[Dict[str, Any]]] = None
    gradcam_overlay: Optional[str] = None
    gradcam_pp_overlay: Optional[str] = None
    disclaimer: str = "Research use only. Not a medical diagnosis."
    model_config = ConfigDict(populate_by_name=True, extra="allow")


# ─── Analysis Summary Schema (For History Table) ────────────────────────
class AnalysisSummaryResponse(BaseModel):
    id: str
    fileName: str
    thumbnailUrl: str
    predictedClass: Optional[str] = None
    confidence: Optional[float] = None
    status: str
    modelVersion: str
    createdAt: str
    isDemo: bool = False
    workflow_type: Optional[str] = "2D_CLASSIFICATION"  # 2D_CLASSIFICATION | 3D_SEGMENTATION
    analysis_mode: Optional[str] = "classification_2d"
    model_id: Optional[str] = None
    model_name: Optional[str] = None
    actual_model: Optional[ActualModelMetadata] = None
