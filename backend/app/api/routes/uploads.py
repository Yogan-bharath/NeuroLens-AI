import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status

from app.models.user import UserDB
from app.api.deps import get_current_user_or_default
from app.utils.file_validation import validate_file_bytes
from app.services.imagekit_service import imagekit_service
from app.schemas.upload import ImageUploadResponse

router = APIRouter(prefix="/uploads", tags=["Uploads"])

# Ephemeral upload registry for tracking uploads before analysis creation
_uploads_registry = {}


@router.post("/mri", response_model=ImageUploadResponse)
async def upload_mri_file(
    file: UploadFile = File(...),
    current_user: UserDB = Depends(get_current_user_or_default)
):
    """
    Upload a brain MRI scan (PNG/JPEG) to ImageKit.
    Validates file signature, dimensions, and MIME type.
    """
    content = await file.read()
    filename = file.filename or "mri_scan.png"

    # Validate
    width, height, mime_type = validate_file_bytes(content, filename)

    # Upload to ImageKit
    upload_res = await imagekit_service.upload_file(
        content, filename, folder=f"neurolens/users/{current_user.id}/uploads"
    )

    upload_id = f"upl_{uuid.uuid4().hex[:12]}"
    record = ImageUploadResponse(
        upload_id=upload_id,
        file_id=upload_res["file_id"],
        url=upload_res["url"],
        filename=filename,
        size_bytes=upload_res["size_bytes"],
        width=width,
        height=height,
        mime_type=mime_type,
        storage_provider=upload_res["storage_provider"],
        created_at=datetime.now(timezone.utc)
    )

    _uploads_registry[upload_id] = {
        "record": record,
        "bytes": content,
        "user_id": current_user.id
    }

    return record


@router.get("/{upload_id}", response_model=ImageUploadResponse)
async def get_upload_record(
    upload_id: str,
    current_user: UserDB = Depends(get_current_user_or_default)
):
    """Retrieve metadata for an existing image upload."""
    entry = _uploads_registry.get(upload_id)
    if not entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Upload with identifier '{upload_id}' was not found."
        )
    return entry["record"]
