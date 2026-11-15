"""
NeuroScan Persistent File Storage Service
Manages local and cloud asset persistence with clear separation:
- uploads: Original 2D MRI scans (PNG, JPG, JPEG)
- heatmaps: Grad-CAM and XAI saliency heatmaps
- slices: Multiplanar cross-sectional previews (Axial, Coronal, Sagittal)
- volumes_raw: Uploaded 3D multimodal NIfTI series (FLAIR, T1, T1CE, T2, SEG)
- volumes_predictions: Serialized segmentation volumes (.nii.gz)
Includes SHA-256 deduplication, atomic writes, and secure URL generation.
"""

import os
import hashlib
import uuid
import mimetypes
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from app.core.config import settings
from app.core.logging import logger

# Base persistent storage directory
STORAGE_BASE_DIR = Path(settings._BASE_DIR) / "storage"

CATEGORIES = {
    "uploads": STORAGE_BASE_DIR / "uploads",
    "heatmaps": STORAGE_BASE_DIR / "heatmaps",
    "slices": STORAGE_BASE_DIR / "slices",
    "volumes_raw": STORAGE_BASE_DIR / "volumes" / "raw",
    "volumes_predictions": STORAGE_BASE_DIR / "volumes" / "predictions",
}

for cat_dir in CATEGORIES.values():
    cat_dir.mkdir(parents=True, exist_ok=True)


class StorageService:
    def __init__(self):
        self.base_dir = STORAGE_BASE_DIR

    def compute_sha256(self, content: bytes) -> str:
        """Computes SHA-256 hash for content deduplication."""
        h = hashlib.sha256()
        h.update(content)
        return h.hexdigest()

    def store_file(
        self,
        category: str,
        content: bytes,
        filename: str,
        custom_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Stores file bytes in the specified category directory.
        Uses content hash or custom_id to ensure deduplication and persistence.
        """
        target_dir = CATEGORIES.get(category)
        if target_dir is None:
            target_dir = self.base_dir / category
            target_dir.mkdir(parents=True, exist_ok=True)

        content_hash = self.compute_sha256(content)
        ext = "".join(Path(filename).suffixes)
        if not ext:
            ext = ".bin"

        file_id = custom_id if custom_id else f"{content_hash[:16]}_{uuid.uuid4().hex[:6]}"
        safe_filename = f"{file_id}{ext}"
        file_path = target_dir / safe_filename

        # Write if not already present
        if not file_path.exists():
            temp_path = file_path.with_suffix(f"{file_path.suffix}.tmp_{uuid.uuid4().hex[:6]}")
            with open(temp_path, "wb") as f:
                f.write(content)
            temp_path.replace(file_path)
            logger.info("Saved persistent file [%s]: %s (%d bytes)", category, file_path.name, len(content))
        else:
            logger.debug("File already exists in storage [%s]: %s", category, file_path.name)

        relative_url = f"/api/storage/{category}/{safe_filename}"
        mime_type, _ = mimetypes.guess_type(filename)
        if not mime_type:
            mime_type = "application/gzip" if ext in [".nii.gz", ".gz"] else "application/octet-stream"

        return {
            "file_id": safe_filename,
            "url": relative_url,
            "path": str(file_path),
            "size_bytes": len(content),
            "content_hash": content_hash,
            "mime_type": mime_type,
            "category": category,
        }

    def get_file_path(self, category: str, filename: str) -> Optional[Path]:
        """Returns the Path to the stored file if it exists, safely preventing directory traversal."""
        target_dir = CATEGORIES.get(category)
        if target_dir is None:
            return None
        safe_name = os.path.basename(filename)
        file_path = target_dir / safe_name
        return file_path if file_path.exists() and file_path.is_file() else None

    def get_file(self, category: str, filename: str) -> Optional[Tuple[bytes, str, str]]:
        """
        Retrieves file bytes, MIME type, and absolute path for a stored asset.
        """
        target_dir = CATEGORIES.get(category)
        if target_dir is None:
            return None

        # Prevent directory traversal
        safe_name = os.path.basename(filename)
        file_path = target_dir / safe_name

        if not file_path.exists() or not file_path.is_file():
            return None

        mime_type, _ = mimetypes.guess_type(safe_name)
        if not mime_type:
            if safe_name.endswith(".nii.gz"):
                mime_type = "application/gzip"
            elif safe_name.endswith(".nii"):
                mime_type = "application/octet-stream"
            elif safe_name.endswith(".png"):
                mime_type = "image/png"
            elif safe_name.endswith(".jpg") or safe_name.endswith(".jpeg"):
                mime_type = "image/jpeg"
            else:
                mime_type = "application/octet-stream"

        with open(file_path, "rb") as f:
            content = f.read()

        return content, mime_type, str(file_path)

    def delete_file(self, category: str, filename: str) -> bool:
        """Deletes a stored file if it exists."""
        target_dir = CATEGORIES.get(category)
        if target_dir is None:
            return False

        safe_name = os.path.basename(filename)
        file_path = target_dir / safe_name
        if file_path.exists():
            try:
                file_path.unlink()
                logger.info("Deleted persistent file [%s]: %s", category, safe_name)
                return True
            except Exception as e:
                logger.error("Failed to delete [%s] %s: %s", category, safe_name, e)
                return False
        return False


storage_service = StorageService()

