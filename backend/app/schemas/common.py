from typing import Generic, List, Optional, TypeVar, Dict, Any
from pydantic import BaseModel, Field

T = TypeVar("T")


class MessageResponse(BaseModel):
    message: str
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    service: str


class ReadinessResponse(BaseModel):
    status: str  # ready | degraded | unavailable
    database_connected: bool
    models_ready: bool
    storage_ready: bool
    details: Dict[str, Any] = Field(default_factory=dict)


class PaginatedResponse(BaseModel, Generic[T]):
    items: List[T]
    total: int
    page: int
    page_size: int
    total_pages: int
