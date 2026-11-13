"""
Storage API Router
Serves persistent assets (uploads, heatmaps, slices, NIfTI volumes) with
streaming responses, proper MIME headers, and caching.
"""

import os
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse, Response
from app.services.storage_service import storage_service, CATEGORIES

router = APIRouter(prefix="/storage", tags=["Storage"])


@router.get("/{category}/{filename}")
async def serve_stored_file(category: str, filename: str):
    """
    Streams a stored file by category and filename.
    Valid categories: uploads, heatmaps, slices, volumes_raw, volumes_predictions.
    """
    if category not in CATEGORIES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid storage category: '{category}'. Valid categories: {list(CATEGORIES.keys())}"
        )

    file_tuple = storage_service.get_file(category, filename)
    if file_tuple is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{filename}' in category '{category}' was not found on host."
        )

    content, mime_type, file_path = file_tuple

    headers = {
        "Cache-Control": "public, max-age=86400",
        "X-Content-Type-Options": "nosniff",
    }

    if category in ["volumes_predictions", "volumes_raw"] or filename.endswith((".nii", ".nii.gz")):
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'

    return FileResponse(
        path=file_path,
        media_type=mime_type,
        headers=headers,
        filename=filename if headers.get("Content-Disposition") else None
    )

