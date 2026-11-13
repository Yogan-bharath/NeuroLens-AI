from typing import Optional
from pydantic import BaseModel, Field


class CopilotChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1200)
    analysis_id: Optional[str] = Field(None, max_length=100)
    conversation_id: Optional[str] = Field(None, max_length=100)


class CopilotChatResponse(BaseModel):
    id: str
    role: str = "assistant"
    content: str
    timestamp: str
    conversation_id: str
    context_analysis_id: Optional[str] = None


class CopilotClearRequest(BaseModel):
    conversation_id: str = Field(..., min_length=1, max_length=100)
