from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class AnalysisDB(BaseModel):
    id: Optional[str] = Field(default=None, alias="_id")
    analysis_id: str
    user_id: str
    original_filename: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "queued"  # queued | processing | completed | failed
    analysis_mode: str = "combined"  # classification | segmentation | combined

    input_image: Dict[str, Any] = Field(default_factory=dict)
    classification: Optional[Dict[str, Any]] = None
    segmentation: Optional[Dict[str, Any]] = None
    explainability: List[Dict[str, Any]] = Field(default_factory=list)
    configuration: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    processing_metadata: Dict[str, Any] = Field(default_factory=dict)
    is_demo: bool = False

    class Config:
        populate_by_name = True
