from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from bson import ObjectId

from app.core.config import settings
from app.core.security import decode_access_token
from app.core.exceptions import AuthenticationException, PermissionDeniedException
from app.db.mongodb import get_database
from app.models.user import UserDB

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_PREFIX}/auth/login", auto_error=False)

# In-memory users for offline/test environments
_in_memory_users: Dict[str, Dict[str, Any]] = {}


async def get_current_user_optional(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme)
) -> Optional[UserDB]:
    """
    Extract current user from Bearer header or HttpOnly cookie.
    Returns None if no token provided.
    """
    # Check Cookie fallback if Bearer header is missing
    if not token:
        token = request.cookies.get("access_token")

    if not token:
        return None

    payload = decode_access_token(token)
    if not payload:
        return None

    user_id = payload.get("sub")
    if not user_id:
        return None

    db = get_database()
    if db is not None:
        try:
            # Query by ObjectId or string ID
            try:
                user_doc = await db.users.find_one({"_id": ObjectId(user_id)})
            except Exception:
                user_doc = await db.users.find_one({"_id": user_id})

            if not user_doc:
                user_doc = await db.users.find_one({"email": payload.get("email")})

            if user_doc:
                user_doc["_id"] = str(user_doc["_id"])
                return UserDB(**user_doc)
        except Exception:
            pass

    # Check in-memory store
    if user_id in _in_memory_users:
        return UserDB(**_in_memory_users[user_id])

    for u in _in_memory_users.values():
        if u.get("email") == payload.get("email"):
            return UserDB(**u)

    # If payload is valid but user was removed, return lightweight UserDB
    return UserDB(
        _id=user_id,
        name=payload.get("name", "Researcher"),
        email=payload.get("email", "researcher@example.com"),
        password_hash="",
        role=payload.get("role", "researcher")
    )


async def get_current_user(
    current_user: Optional[UserDB] = Depends(get_current_user_optional)
) -> UserDB:
    """Enforce authenticated user requirement."""
    if not current_user:
        raise AuthenticationException("Authentication required. Please log in.")
    if not current_user.is_active:
        raise AuthenticationException("User account is disabled.")
    return current_user


async def get_current_user_or_default(
    current_user: Optional[UserDB] = Depends(get_current_user_optional)
) -> UserDB:
    """
    Returns authenticated user if logged in,
    otherwise returns a default demo researcher context for unauthenticated usage.
    """
    if current_user:
        return current_user
    return UserDB(
        _id="guest_researcher",
        name="Guest Researcher",
        email="guest@example.com",
        password_hash="",
        role="researcher"
    )


async def get_current_admin(
    current_user: UserDB = Depends(get_current_user)
) -> UserDB:
    """Enforce administrator role requirement."""
    if current_user.role != "admin":
        raise PermissionDeniedException("Administrator role required.")
    return current_user
