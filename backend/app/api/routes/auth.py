from datetime import datetime, timezone
from fastapi import APIRouter, Depends, Response, HTTPException, status
from bson import ObjectId

from app.core.config import settings
from app.core.security import get_password_hash, verify_password, create_access_token
from app.core.exceptions import AuthenticationException
from app.db.mongodb import get_database
from app.models.user import UserDB
from app.schemas.auth import UserRegisterRequest, UserLoginRequest, UserResponse, TokenResponse
from app.api.deps import get_current_user, _in_memory_users

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=TokenResponse)
async def register_user(request: UserRegisterRequest, response: Response):
    """Register a new research user account."""
    normalized_email = request.email.lower().strip()
    db = get_database()

    # Check existing user
    if db is not None:
        try:
            existing = await db.users.find_one({"email": normalized_email})
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="A user with this email address already exists."
                )
        except HTTPException:
            raise
        except Exception:
            pass

    if normalized_email in [u.get("email") for u in _in_memory_users.values()]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A user with this email address already exists."
        )

    # Hash password
    pwd_hash = get_password_hash(request.password)
    now = datetime.now(timezone.utc)

    user_dict = {
        "name": request.name,
        "email": normalized_email,
        "password_hash": pwd_hash,
        "role": request.role or "researcher",
        "is_active": True,
        "created_at": now,
        "updated_at": now,
    }

    user_id = None
    if db is not None:
        try:
            res = await db.users.insert_one(user_dict)
            user_id = str(res.inserted_id)
        except Exception:
            pass

    if not user_id:
        user_id = f"usr_{len(_in_memory_users) + 1}"
        user_dict["_id"] = user_id
        _in_memory_users[user_id] = user_dict

    # Create JWT
    token = create_access_token(
        subject=user_id,
        extra_claims={"email": normalized_email, "name": request.name, "role": user_dict["role"]}
    )

    # Set secure HttpOnly cookie
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
        secure=settings.is_production
    )

    user_res = UserResponse(
        id=user_id,
        name=request.name,
        email=normalized_email,
        role=user_dict["role"],
        is_active=True,
        created_at=now
    )

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=user_res
    )


@router.post("/login", response_model=TokenResponse)
async def login_user(request: UserLoginRequest, response: Response):
    """Authenticate with email and password."""
    normalized_email = request.email.lower().strip()
    db = get_database()
    user_doc = None

    if db is not None:
        try:
            user_doc = await db.users.find_one({"email": normalized_email})
        except Exception:
            pass

    if not user_doc:
        for u in _in_memory_users.values():
            if u.get("email") == normalized_email:
                user_doc = u
                break

    if not user_doc:
        raise AuthenticationException("Invalid email or password.")

    if not verify_password(request.password, user_doc.get("password_hash", "")):
        raise AuthenticationException("Invalid email or password.")

    user_id = str(user_doc.get("_id"))
    token = create_access_token(
        subject=user_id,
        extra_claims={
            "email": user_doc["email"],
            "name": user_doc["name"],
            "role": user_doc.get("role", "researcher")
        }
    )

    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
        secure=settings.is_production
    )

    user_res = UserResponse(
        id=user_id,
        name=user_doc["name"],
        email=user_doc["email"],
        role=user_doc.get("role", "researcher"),
        is_active=user_doc.get("is_active", True),
        created_at=user_doc.get("created_at", datetime.now(timezone.utc))
    )

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=user_res
    )


@router.post("/logout")
async def logout_user(response: Response):
    """Clear session authentication cookies."""
    response.delete_cookie(key="access_token")
    return {"message": "Logged out successfully"}


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: UserDB = Depends(get_current_user)):
    """Retrieve currently authenticated user profile."""
    return UserResponse(
        id=current_user.id or "usr_unknown",
        name=current_user.name,
        email=current_user.email,
        role=current_user.role,
        is_active=current_user.is_active,
        created_at=current_user.created_at
    )
