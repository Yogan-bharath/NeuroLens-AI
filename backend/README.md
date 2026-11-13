# NeuroLens AI — Production Backend

A high-performance, modular backend for **NeuroLens AI**, a brain MRI tumor classification, segmentation, and explainable AI (XAI) research platform.

Built with **FastAPI**, **MongoDB Atlas**, **ImageKit**, **Pydantic v2**, and **TensorFlow / Keras**.

---

## 1. Architectural Overview

```
                      +---------------------------------------+
                      |       React + Vite Frontend           |
                      |   (Theme, Wizard, Workspace, Chat)    |
                      +-------------------+-------------------+
                                          | REST API (HTTP / JSON / Multipart)
                                          v
                      +---------------------------------------+
                      |        FastAPI Gateway & Router       |
                      |  - JWT Authentication & RBAC          |
                      |  - CORS & Secure Cookie Protection    |
                      |  - Multipart MRI File Validation      |
                      +-------------------+-------------------+
                                          |
        +---------------------------------+---------------------------------+
        |                                 |                                 |
        v                                 v                                 v
+----------------+               +-----------------+               +-----------------+
|  MongoDB Atlas |               | ImageKit Storage|               | ML Service Hub  |
| - Users        |               | - Original Scans|               | - EfficientNet  |
| - Analyses     |               | - Grad-CAM Maps |               | - U-Net Segment |
| - Models Reg   |               | - Saliency Blends|              | - XAI Grad-CAM++|
+----------------+               +-----------------+               +-----------------+
```

---

## 2. Model Integration Strategy

Per project constraints, trained model files are stored in external artifact storage (e.g. Google Drive) and may not be initially present in the repository:

1. **Clean Service Interfaces**:
   - `ClassificationService` (`app/services/classification_service.py`)
   - `SegmentationService` (`app/services/segmentation_service.py`)
   - `ExplainabilityService` (`app/services/explainability_service.py`)
   - `ModelLoader` (`app/ml/model_loader.py`)

2. **Explicit Development Mock vs. Production Separation**:
   - **Development Mode (`ENVIRONMENT=development`)**:
     When model weights are missing, the system uses an explicitly labeled dev mock service with `is_mock=True` / `isDemo=True`. Responses are clearly identified as non-clinical development fixtures and never masquerade as real clinical inferences.
   - **Production Mode (`ENVIRONMENT=production`)**:
     If weights are absent, endpoints immediately reject requests with **`503 Service Unavailable`** (`MODEL_UNAVAILABLE`) with clear diagnostic reasons, preventing any fabricated outputs in a production environment.

3. **Configuring Real Weights**:
   Place trained weights in `backend/model_artifacts/`:
   ```bash
   backend/model_artifacts/EfficientNetV2B0_clahe_iid_best.keras
   backend/model_artifacts/unet_segmentation_best.keras
   ```
   The backend automatically discovers, validates input/output tensors, registers convolutional target layers, and runs hardware-accelerated inference.

   **Attention CNN + ViT is not enabled for inference.** The available artifact has a 2D `(128, 128, 4)` input and four-channel softmax output, but its matching training preprocessing and output-channel mapping have not been verified. A separate 3D U-Net patch pipeline is not compatible with this artifact. The model remains listed as coming soon, and requests for it return `503 MODEL_UNAVAILABLE` until its own training materials are verified.

   **BraTS 3D U-Net integration:** the checked training notebook verifies FLAIR/T1/T1CE/T2 ordering, independent non-zero-voxel z-score normalization, 96 × 96 × 64 patches, (64, 64, 32) sliding-window stride, softmax/argmax output, and label remapping (0→0, 1→1, 2→2, 4→3). The backend pipeline is prepared at `app/services/brats_3d_service.py`; install the separately trained `best_3d_unet_model.keras` at `backend/model_artifacts/brats_3d_unet/` to load and inference-validate it. Until that checkpoint exists and passes shape and smoke-inference checks, the model catalog keeps it unavailable and `POST /api/segmentation/brats-3d-unet` returns `503 MODEL_UNAVAILABLE`. The authenticated inference route accepts four aligned NIfTI files, runs segmentation only, exports original BraTS labels (including enhancing tumor as label 4), and does not upload source scans to ImageKit. This is a research tool, not a medical diagnosis.

---

## 3. Technology Stack

- **Python**: 3.11.9
- **Framework**: FastAPI (ASGI)
- **Server**: Uvicorn with lifespan management
- **Database**: MongoDB Atlas via Motor (Async PyMongo)
- **Storage**: ImageKit Python SDK (`imagekitio`)
- **Validation**: Pydantic v2 + `pydantic-settings`
- **Security**: JWT (HS256) + bcrypt password hashing + HttpOnly Cookies
- **Computer Vision**: OpenCV Headless, Pillow, NumPy, Scikit-learn
- **Reporting**: ReportLab (Vector PDF Generation)
- **Testing**: Pytest, Pytest-asyncio, HTTPX

---

## 4. API Endpoints Reference

### Health & Readiness
- `GET /api/health` — Service liveness probe.
- `GET /api/health/ready` — Readiness probe checking database, storage, and models.

### Authentication & Authorization
- `POST /api/auth/register` — Create researcher account with bcrypt-hashed credentials.
- `POST /api/auth/login` — Authenticate and receive JWT token + secure cookie.
- `POST /api/auth/logout` — Revoke session cookies.
- `GET /api/auth/me` — Authenticated profile.

### Models & Specifications
- `GET /api/models` — Full model and XAI registry.
- `GET /api/models/classification` — Classification models (EfficientNetV2B0, ResNet50, DenseNet121).
- `GET /api/models/segmentation` — Segmentation models (U-Net, Attention U-Net, U-Net++).
- `GET /api/models/xai-methods` — Compatible explainability methods and verified target layers.
- `GET /api/models/compare` — Cross-model benchmark comparison matrix.
- `GET /api/models/{model_id}` — Architectural layer maps and verified metrics.

### Scans & Analysis
- `POST /api/uploads/mri` — Upload MRI scan (PNG/JPEG) to ImageKit with dimension checks.
- `POST /api/analyses` — Create and run analysis (accepts Multipart or JSON base64).
- `GET /api/analyses/{id}` — Full analysis workspace result.
- `GET /api/analyses/{id}/status` — Analysis pipeline progress check.
- `POST /api/analyses/{id}/retry` — Re-execute processing pipeline.
- `DELETE /api/analyses/{id}` — Delete analysis and associated ImageKit artifacts.

### History
- `GET /api/history` — Paginated analysis history with search and status/class filters.
- `GET /api/history/stats` — User analysis volume and class distributions.

### Explainability (XAI)
- `POST /api/analyses/{id}/explain` — Compute on-demand Grad-CAM, Grad-CAM++, or Seg Grad-CAM.
- `GET /api/analyses/{id}/explanations` — Retrieve all generated attribution heatmaps.

### Performance & Metrics
- `GET /api/performance/summary` — Accuracy, macro metrics, confusion matrix.
- `GET /api/performance/classification` — Per-class precision, recall, and F1.
- `GET /api/performance/segmentation` — Verified ground-truth Dice (0.884) & IoU (0.792).

### Clinical Research Reports
- `GET /api/analyses/{id}/report` — Summary metadata.
- `GET /api/analyses/{id}/report/download` — Download research PDF report.

### Frontend Compatibility Routes (`/api/v1`)
- `POST /api/v1/analyze` — Maps directly to `create_analysis`.
- `GET /api/v1/analysis/{id}` — Maps directly to `get_analysis_by_id`.
- `GET /api/v1/history` — Maps directly to `get_history_list`.
- `GET /api/v1/model/metrics` — Maps directly to `get_performance_summary`.
- `POST /api/v1/chat` — NeuroLens AI Copilot natural language responses.

---

## 5. Local Setup & Execution

### 1. Environment Preparation
```bash
cd backend
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration (`.env`)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Populate your credentials:
```env
MONGODB_URL="mongodb+srv://<user>:<password>@cluster0.mongodb.net/?retryWrites=true&w=majority"
MONGODB_DB_NAME="neurolens_ai"
IMAGEKIT_PUBLIC_KEY="public_..."
IMAGEKIT_PRIVATE_KEY="private_..."
IMAGEKIT_URL_ENDPOINT="https://ik.imagekit.io/your_id"
```

### 3. Launch the Backend Server
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Interactive API documentation: `http://localhost:8000/docs`

### 4. Run Automated Tests
```bash
pytest tests -v
```

---

## 6. Vercel Feasibility & Deployment Assessment

### Vercel Serverless Feasibility
- **Feasible Components**: Auth, User Management, MongoDB queries, ImageKit uploads, History, Reports, and API Gateway.
- **Constraints**: Vercel Serverless functions have an uncompressed bundle size limit of **250MB** and maximum duration of **10s - 60s**. The complete TensorFlow / Keras deep learning package exceeds 500MB, making unified TensorFlow deployment on Vercel unadvisable for heavy inference.

### Recommended Production Deployment Architectures
1. **Unified Container (Recommended)**:
   Deploy FastAPI via Docker to **Render**, **Railway**, **Fly.io**, or **AWS ECS / App Runner** with 2GB+ RAM. Both API and ML inference run efficiently in a single persistent container.
2. **Decoupled Architecture**:
   - Vercel: Hosts React frontend and API routing.
   - Dedicated GPU/CPU Inference Worker: FastAPI ML service hosting TensorFlow models and Grad-CAM computation.

---

## 7. Implementation Status

| Feature | Status | Notes |
|---|---|---|
| FastAPI Foundation & Lifespan | **Implemented & Tested** | Lifespan events, logging, structured exceptions |
| MongoDB Atlas Integration | **Implemented & Tested** | Motor async driver, unique indexes, fallback store |
| ImageKit Storage Service | **Implemented & Tested** | Per-analysis folders, server-generated IDs, dev fallback |
| JWT Authentication & Security | **Implemented & Tested** | Bcrypt hashing, HttpOnly cookies, RBAC, user isolation |
| Uploads & Preprocessing | **Implemented & Tested** | Magic byte validation, CLAHE, resizing, colormapping |
| Classification Interface | **Implemented & Tested** | 4-class pipeline, clean interface for weights |
| Segmentation Interface | **Implemented & Tested** | Multiclass mask overlays, ground-truth only Dice/IoU |
| Explainability Service | **Implemented & Tested** | Grad-CAM, Grad-CAM++, layer validation, ImageKit uploads |
| Downloadable PDF Reports | **Implemented & Tested** | ReportLab generation with research use disclaimers |
| AI Research Chat | **Implemented & Tested** | Conversational assistance for MRI/XAI questions |
| Frontend Contract Compatibility | **Implemented & Tested** | Both `/api` and `/api/v1` routes fully supported |
| Automated Pytest Suite | **Implemented & Tested** | 10/10 test modules passing with 0 warnings |
