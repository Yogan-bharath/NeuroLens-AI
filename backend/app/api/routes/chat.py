from fastapi import APIRouter, Depends
from typing import Optional

from app.schemas.chat import ChatRequest, ChatMessageResponse
from app.services.copilot_service import copilot_service
from app.models.user import UserDB
from app.api.deps import get_current_user_optional

router = APIRouter(tags=["AI Research Chat"])


@router.post("/chat", response_model=ChatMessageResponse)
@router.post("/v1/chat", response_model=ChatMessageResponse)
async def handle_chat_message(
    request: ChatRequest,
    current_user: Optional[UserDB] = Depends(get_current_user_optional)
):
    """
    Context-aware Copilot for natural language inquiries regarding MRI classification,
    saliency heatmaps, and binary U-Net segmentation.
    Grounded in MongoDB analysis record when analysis_id is provided.
    """
    user_id = str(current_user.id) if current_user else "guest_researcher"
    return await copilot_service.chat(request.message, request.analysis_id, None, user_id)
