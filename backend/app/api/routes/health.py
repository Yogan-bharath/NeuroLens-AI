from fastapi import APIRouter
from app.core.config import settings
from app.db.mongodb import get_database
from app.ml.model_loader import model_loader
from app.services.imagekit_service import imagekit_service
from app.schemas.common import HealthResponse, ReadinessResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def get_health():
    """Liveness probe endpoint."""
    return HealthResponse(
        status="ok",
        version="v2.4.0",
        environment=settings.ENVIRONMENT,
        service="NeuroScan Backend"
    )


@router.get("/health/ready", response_model=ReadinessResponse)
async def get_readiness():
    """Readiness probe checking database, storage, and models."""
    db = get_database()
    db_connected = False
    if db is not None:
        try:
            await db.client.admin.command("ping")
            db_connected = True
        except Exception:
            db_connected = False

    model_status = model_loader.get_status()
    models_ready = (
        model_status["classification"]["available"]
        or model_status["segmentation"]["available"]
        or (settings.is_dev and settings.USE_MOCK_MODELS_IN_DEV)
    )

    storage_ready = imagekit_service.is_live or settings.is_dev

    overall_status = "ready"
    if not db_connected and settings.is_production:
        overall_status = "degraded"
    elif not models_ready and settings.is_production:
        overall_status = "unavailable"

    return ReadinessResponse(
        status=overall_status,
        database_connected=db_connected,
        models_ready=models_ready,
        storage_ready=storage_ready,
        details={
            "classification_loaded": model_status["classification"]["available"],
            "segmentation_loaded": model_status["segmentation"]["available"],
            "storage_provider": "imagekit" if imagekit_service.is_live else "local_dev",
            "environment": settings.ENVIRONMENT
        }
    )
