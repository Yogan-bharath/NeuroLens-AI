from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class ImageUploadResponse(BaseModel):
    upload_id: str
    file_id: str
    url: str
    filename: str
    size_bytes: int
    width: int
    height: int
    mime_type: str
    storage_provider: str
    created_at: datetime
