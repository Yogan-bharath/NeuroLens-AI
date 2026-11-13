from fastapi import APIRouter, Depends

from app.api.deps import get_current_user_or_default
from app.core.config import settings
from app.models.user import UserDB
from app.schemas.copilot import CopilotChatRequest, CopilotChatResponse, CopilotClearRequest
from app.services.copilot_service import copilot_service

router = APIRouter(prefix="/copilot", tags=["AI Research Copilot"])


@router.post("/chat", response_model=CopilotChatResponse)
async def chat(request: CopilotChatRequest, current_user: UserDB = Depends(get_current_user_or_default)):
    return await copilot_service.chat(request.message, request.analysis_id, request.conversation_id, str(current_user.id))


@router.post("/clear-history")
async def clear_history(request: CopilotClearRequest, current_user: UserDB = Depends(get_current_user_or_default)):
    copilot_service.clear(request.conversation_id, str(current_user.id))
    return {"cleared": True, "conversation_id": request.conversation_id}


@router.get("/health")
async def health():
    return {"status": "ready", "provider": "groq", "configured": bool(settings.GROQ_API_KEY)}
