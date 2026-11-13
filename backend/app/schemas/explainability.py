from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class GenerateExplanationRequest(BaseModel):
    method: str = Field(..., description="gradcam | gradcam_pp | integrated_gradients | seg_gradcam | mask_guided_cam")
    target_layer: Optional[str] = None
    target_class: Optional[str] = None
    target_region: Optional[str] = None


class ExplanationArtifactResponse(BaseModel):
    analysis_id: str
    method: str
    target_layer: Optional[str] = None
    target_class: Optional[str] = None
    target_region: Optional[str] = None
    heatmap_url: str
    overlay_url: str
    description: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
