import io
from typing import Tuple
from PIL import Image
from fastapi import UploadFile
from app.core.config import settings
from app.core.exceptions import InvalidConfigurationException

# Magic signatures for image formats
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"


def validate_file_bytes(content: bytes, filename: str) -> Tuple[int, int, str]:
    """
    Validate image file size, magic header bytes, and extract dimensions.
    Returns (width, height, mime_type).
    """
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(content) > max_bytes:
        raise InvalidConfigurationException(
            f"File size exceeds maximum allowed limit of {settings.MAX_UPLOAD_SIZE_MB}MB."
        )

    # Magic byte check
    is_png = content.startswith(PNG_MAGIC)
    is_jpeg = content.startswith(JPEG_MAGIC)

    if not (is_png or is_jpeg):
        raise InvalidConfigurationException(
            "Invalid file signature. Only verified PNG and JPEG MRI scans are permitted."
        )

    mime_type = "image/png" if is_png else "image/jpeg"

    # PIL verification
    try:
        with Image.open(io.BytesIO(content)) as img:
            img.verify()
            width, height = img.size
    except Exception as e:
        raise InvalidConfigurationException(f"Corrupted or unreadable image file: {e}")

    # Re-open to confirm dimensions (verify() might close image)
    with Image.open(io.BytesIO(content)) as img:
        width, height = img.size

    if width < 32 or height < 32:
        raise InvalidConfigurationException("Image dimensions are too small for MRI processing (min 32x32).")

    return width, height, mime_type
