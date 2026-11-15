from typing import List, Union
from pydantic import AnyHttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    PROJECT_NAME: str = "NeuroLens AI API"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    API_PREFIX: str = "/api"

    # Security
    SECRET_KEY: str = "neurolens-secret-key-change-in-production-min-32-chars-long"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours

    # CORS
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, list):
            return v
        return ["http://localhost:5173", "http://127.0.0.1:5173"]

    # MongoDB Atlas
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "neurolens_ai"
    MONGODB_TIMEOUT_MS: int = 5000

    # ImageKit
    IMAGEKIT_PUBLIC_KEY: str = ""
    IMAGEKIT_PRIVATE_KEY: str = ""
    IMAGEKIT_URL_ENDPOINT: str = ""

    # ML Artifacts & Inference Pipeline
    _BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _DEFAULT_MODEL_DIR = os.path.join(_BASE_DIR, "model_artifacts") if os.path.exists(os.path.join(_BASE_DIR, "model_artifacts")) else "./model_artifacts"

    MODEL_DIR: str = _DEFAULT_MODEL_DIR
    DEFAULT_CLASSIFIER: str = "efficientnetv2b0_256"
    ENABLE_LEGACY_MODELS: bool = True
    ENABLE_SEG_XAI: bool = True
    ENABLE_3D_UNET: bool = True
    CLASSIFICATION_MODEL_PATH: str = os.path.join(_DEFAULT_MODEL_DIR, "EfficientNetV2B0_clahe_iid_best.keras")
    CLASSIFICATION_MODEL_256_PATH: str = os.path.join(_DEFAULT_MODEL_DIR, "EfficientNetV2B0_clahe_256.keras")
    CLASSIFICATION_MODEL_LEGACY_PATH: str = os.path.join(_DEFAULT_MODEL_DIR, "EfficientNetV2B0_clahe_iid_best.keras")
    SEGMENTATION_MODEL_PATH: str = os.path.join(_DEFAULT_MODEL_DIR, "BRISC_Binary_UNet_best.keras")
    DEEPLAB_MODEL_PATH: str = os.path.join(_DEFAULT_MODEL_DIR, "deeplabv3plus.keras")
    BRATS_3D_UNET_MODEL_PATH: str = os.path.join(_DEFAULT_MODEL_DIR, "brats_3d_unet", "best_3d_unet_model.keras")
    USE_MOCK_MODELS_IN_DEV: bool = True
    IMG_SIZE: int = 128
    SEGMENTATION_THRESHOLD: float = 0.5
    DEEPLAB_INPUT_NORMALIZATION: str = "zero_one"  # zero_one | none; confirm with training metadata
    CLASSIFIER_USE_CLAHE: bool = False
    CLASS_MAPPING_CONFIRMED: bool = False
    CLASS_NAMES: List[str] = ["glioma", "meningioma", "pituitary", "no_tumor"]
    SWIN_CLASS_NAMES: List[str] = ["glioma", "meningioma", "pituitary", "no_tumor"]
    SWIN_CLASS_MAPPING_CONFIRMED: bool = False

    # Copilot / LLM Configuration
    LLM_PROVIDER: str = "mock"  # mock | gemini | openai
    LLM_API_KEY: str = ""
    LLM_MODEL_NAME: str = "gemini-1.5-flash"
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GROQ_FALLBACK_MODEL: str = "openai/gpt-oss-20b"
    COPILOT_TIMEOUT_SECONDS: float = 20.0
    COPILOT_MAX_HISTORY_MESSAGES: int = 10
    COPILOT_RATE_LIMIT_PER_MINUTE: int = 12

    # Uploads
    MAX_UPLOAD_SIZE_MB: int = 15
    ALLOWED_EXTENSIONS: str = "png,jpg,jpeg"

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"

    @property
    def is_dev(self) -> bool:
        return self.ENVIRONMENT.lower() == "development"

    @property
    def allowed_extensions_list(self) -> List[str]:
        return [ext.strip().lower().lstrip(".") for ext in self.ALLOWED_EXTENSIONS.split(",")]


settings = Settings()
