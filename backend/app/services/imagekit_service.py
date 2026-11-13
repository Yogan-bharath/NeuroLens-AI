import io
import uuid
import base64
from typing import Optional, Dict, Any
from imagekitio import ImageKit
from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import StorageException


class ImageKitService:
    def __init__(self):
        self._ik: Optional[ImageKit] = None
        self._is_configured = False
        self._init_client()

    def _init_client(self):
        if (
            settings.IMAGEKIT_PRIVATE_KEY
            and settings.IMAGEKIT_URL_ENDPOINT
            and not settings.IMAGEKIT_PRIVATE_KEY.startswith("private_placeholder")
        ):
            try:
                # ImageKit Python SDK v5 initialization
                self._ik = ImageKit(
                    private_key=settings.IMAGEKIT_PRIVATE_KEY
                )
                self._is_configured = True
                logger.info("ImageKit client initialized with live credentials.")
            except Exception as e:
                logger.warning(f"Failed to initialize ImageKit client: {e}")
                self._is_configured = False
        else:
            logger.info("ImageKit is running in local/fallback mode (no live ImageKit keys provided).")

    @property
    def is_live(self) -> bool:
        return self._is_configured and self._ik is not None

    async def upload_file(
        self,
        file_bytes: bytes,
        file_name: str,
        folder: str = "/neurolens/uploads",
        tags: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Upload file to ImageKit or generate deterministic development storage URL.
        """
        unique_name = f"{uuid.uuid4().hex}_{file_name}"
        if self.is_live:
            try:
                # Normalize folder to start with /
                clean_folder = folder if folder.startswith("/") else f"/{folder}"
                upload_res = self._ik.files.upload(
                    file=file_bytes,
                    file_name=unique_name,
                    folder=clean_folder,
                    tags=tags or ["neurolens", "mri"],
                    use_unique_file_name=True,
                )
                
                file_id = getattr(upload_res, "file_id", None) or f"ik_{uuid.uuid4().hex[:12]}"
                url = getattr(upload_res, "url", None) or f"{settings.IMAGEKIT_URL_ENDPOINT}{clean_folder}/{unique_name}"

                return {
                    "file_id": file_id,
                    "url": url,
                    "file_name": unique_name,
                    "size_bytes": len(file_bytes),
                    "storage_provider": "imagekit"
                }
            except Exception as e:
                logger.error(f"ImageKit upload failed: {e}")
                if settings.is_production:
                    raise StorageException(f"ImageKit storage upload failed: {str(e)}")

        # Persistent local storage fallback
        from app.services.storage_service import storage_service
        stored = storage_service.store_file("uploads", file_bytes, file_name)

        return {
            "file_id": stored["file_id"],
            "url": stored["url"],
            "file_name": stored["file_id"],
            "size_bytes": stored["size_bytes"],
            "storage_provider": "local_persistent"
        }

    async def delete_file(self, file_id: str) -> bool:
        """Delete file from ImageKit."""
        if not self.is_live or file_id.startswith("dev_"):
            return True
        try:
            self._ik.files.delete(file_id=file_id)
            logger.info(f"Deleted ImageKit file: {file_id}")
            return True
        except Exception as e:
            logger.warning(f"Could not delete ImageKit file {file_id}: {e}")
            return False

    def get_auth_params(self) -> Dict[str, Any]:
        """Generate client-side direct upload parameters."""
        if not self.is_live:
            return {"token": "dev_token", "expire": 0, "signature": "dev_sig"}
        try:
            if hasattr(self._ik, "helper") and hasattr(self._ik.helper, "get_authentication_parameters"):
                return self._ik.helper.get_authentication_parameters()
            return {"token": str(uuid.uuid4()), "expire": 0, "signature": "ik_sig"}
        except Exception as e:
            logger.error(f"Failed to generate ImageKit auth params: {e}")
            return {}


imagekit_service = ImageKitService()
