from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.core.config import settings
from app.core.logging import setup_logging, logger
from app.core.exceptions import NeuroScanException
from app.db.mongodb import connect_to_mongo, close_mongo_connection
from app.db.indexes import create_indexes
from app.ml.model_loader import model_loader

# Import Routers
from app.api.routes import (
    health, auth, users, models, uploads, analyses, history, performance,
    explainability, reports, chat, copilot, segmentation, classification, storage
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    setup_logging()
    logger.info(f"Starting {settings.PROJECT_NAME} in '{settings.ENVIRONMENT}' environment...")
    await connect_to_mongo()
    await create_indexes()
    model_loader.load_models()
    yield
    # Shutdown
    logger.info("Shutting down NeuroScan Backend...")
    await close_mongo_connection()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="2.4.0",
    description="Brain MRI Tumor Classification, Segmentation & Explainable AI Research Platform",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# ─── CORS Configuration ───────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "X-Segmentation-Metadata"],
)

# ─── Global Exception Handlers ────────────────────────────────────────────
@app.exception_handler(NeuroScanException)
async def neuroscan_exception_handler(request: Request, exc: NeuroScanException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": exc.error_code,
            "detail": exc.detail,
            "extra": exc.extra
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = []
    for err in exc.errors():
        loc = " -> ".join([str(x) for x in err.get("loc", [])])
        errors.append(f"{loc}: {err.get('msg', 'validation error')}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error_code": "VALIDATION_ERROR",
            "detail": "Invalid request parameters",
            "errors": errors
        }
    )


# ─── Register Core API Routers Under /api ──────────────────────────────────
api_prefix = settings.API_PREFIX

app.include_router(health.router, prefix=api_prefix)
app.include_router(auth.router, prefix=api_prefix)
app.include_router(users.router, prefix=api_prefix)
app.include_router(models.router, prefix=api_prefix)
app.include_router(uploads.router, prefix=api_prefix)
app.include_router(analyses.router, prefix=api_prefix)
app.include_router(history.router, prefix=api_prefix)
app.include_router(performance.router, prefix=api_prefix)
app.include_router(explainability.router, prefix=api_prefix)
app.include_router(reports.router, prefix=api_prefix)
app.include_router(chat.router, prefix=api_prefix)
app.include_router(copilot.router, prefix=api_prefix)
app.include_router(segmentation.router, prefix=api_prefix)
app.include_router(classification.router, prefix=api_prefix)
app.include_router(storage.router, prefix=api_prefix)

# ─── Frontend Compatibility Layer (/api/v1 routes) ─────────────────────────
# Maps routes expected by existing frontend api.ts directly to backend services
app.add_api_route("/api/v1/analyze", analyses.create_analysis, methods=["POST"], tags=["Frontend V1 Compatibility"])
app.add_api_route("/api/v1/analysis/{analysis_id}", analyses.get_analysis_by_id, methods=["GET"], tags=["Frontend V1 Compatibility"])
app.add_api_route("/api/v1/analysis/{analysis_id}", analyses.delete_analysis_record, methods=["DELETE"], tags=["Frontend V1 Compatibility"])
app.add_api_route("/api/v1/history", history.get_history_list, methods=["GET"], tags=["Frontend V1 Compatibility"])
app.add_api_route("/api/v1/history/{analysis_id}", history.delete_history_item, methods=["DELETE"], tags=["Frontend V1 Compatibility"])
app.add_api_route("/api/v1/model/metrics", performance.get_performance_summary, methods=["GET"], tags=["Frontend V1 Compatibility"])
app.add_api_route("/api/v1/chat", chat.handle_chat_message, methods=["POST"], tags=["Frontend V1 Compatibility"])


@app.get("/")
async def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": "v2.4.0",
        "docs": "/docs",
        "health": f"{settings.API_PREFIX}/health"
    }
