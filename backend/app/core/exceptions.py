from fastapi import HTTPException, status
from typing import Optional, Any, Dict


class NeuroScanException(HTTPException):
    def __init__(
        self,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail: str = "Internal server error",
        error_code: str = "INTERNAL_ERROR",
        extra: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code
        self.extra = extra or {}


class AuthenticationException(NeuroScanException):
    def __init__(self, detail: str = "Could not validate credentials"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            error_code="AUTHENTICATION_FAILED",
        )


class PermissionDeniedException(NeuroScanException):
    def __init__(self, detail: str = "Not authorized to access this resource"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            error_code="PERMISSION_DENIED",
        )


class ResourceNotFoundException(NeuroScanException):
    def __init__(self, resource: str, identifier: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{resource} with ID '{identifier}' was not found",
            error_code="RESOURCE_NOT_FOUND",
        )


class InvalidConfigurationException(NeuroScanException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            error_code="INVALID_CONFIGURATION",
        )


class ModelUnavailableException(NeuroScanException):
    def __init__(self, model_id: str, reason: str = "Model weights or runtime not loaded"):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model '{model_id}' is unavailable: {reason}",
            error_code="MODEL_UNAVAILABLE",
        )


class StorageException(NeuroScanException):
    def __init__(self, detail: str = "Storage service failure"):
        super().__init__(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=detail,
            error_code="STORAGE_FAILURE",
        )
