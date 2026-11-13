from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class FeatureLayerOption(BaseModel):
    id: str
    name: str
    description: str
    isVerified: bool = True
    depthDescription: Optional[str] = None


class ClassificationMetrics(BaseModel):
    testAccuracy: Optional[float] = None
    macroPrecision: Optional[float] = None
    macroRecall: Optional[float] = None
    macroF1: Optional[float] = None
    inferenceTimeMs: Optional[int] = None
    evaluationDataset: Optional[str] = None


class ClassificationModelInfo(BaseModel):
    id: str
    name: str
    architecture: str
    description: str
    task: str = "classification"
    version: str
    status: str  # available | coming_soon | unavailable
    statusReason: Optional[str] = None
    metrics: Optional[ClassificationMetrics] = None
    supportedXaiMethods: List[str] = Field(default_factory=list)
    compatibleLayers: List[FeatureLayerOption] = Field(default_factory=list)
    outputClasses: List[str] = Field(default_factory=list)
    parametersCount: Optional[str] = None


class SegmentationMetrics(BaseModel):
    diceCoefficient: Optional[float] = None
    iouScore: Optional[float] = None
    sensitivity: Optional[float] = None
    inferenceTimeMs: Optional[int] = None
    evaluationDataset: Optional[str] = None


class SegmentationModelInfo(BaseModel):
    id: str
    name: str
    architecture: str
    description: str
    task: str = "Multiclass Semantic Segmentation"
    version: Optional[str] = None
    status: str
    statusReason: Optional[str] = None
    metrics: Optional[SegmentationMetrics] = None
    classes: List[str] = Field(default_factory=list)
    parametersCount: Optional[str] = None
    supportedXaiMethods: List[str] = Field(default_factory=list)
    compatibleLayers: List[FeatureLayerOption] = Field(default_factory=list)


class XaiMethodInfo(BaseModel):
    id: str
    name: str
    description: str
    task: str  # classification | segmentation | both
    status: str
    statusReason: Optional[str] = None
    supportsLayerSelection: bool
    supportedModels: List[str] = Field(default_factory=list)
    type: str  # gradient_activation | path_integral | mask_saliency
