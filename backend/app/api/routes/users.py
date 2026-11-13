from fastapi import APIRouter, Depends
from app.models.user import UserDB
from app.schemas.auth import UserResponse
from app.api.deps import get_current_user

router = APIRouter(prefix="/users", tags=["Users"])


@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: UserDB = Depends(get_current_user)):
    """Get current user details."""
    return UserResponse(
        id=current_user.id or "usr_unknown",
        name=current_user.name,
        email=current_user.email,
        role=current_user.role,
        is_active=current_user.is_active,
        created_at=current_user.created_at
    )
