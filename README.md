# NeuroLens AI

NeuroLens AI is a research platform for brain MRI classification, lesion segmentation, explainability, and analysis review. It pairs a React web app with a FastAPI service and optional TensorFlow/Keras inference models.

> **Research use only.** Results are not a medical diagnosis and must not be used as a substitute for review by a qualified clinician.

## Architecture

```text
Browser
  React 19 + TypeScript + Vite (frontend, port 5173)
                  | HTTP/JSON and multipart uploads
                  v
  FastAPI (backend, port 8000)
    ├── API routes, validation, authentication, and analysis workflows
    ├── Services for classification, segmentation, XAI, reports, and chat
    ├── Model loader and TensorFlow/Keras inference (when installed)
    └── MongoDB for application data
          └── ImageKit for scan and generated-image storage (optional config)
```

The backend is organized around API routes, services, and ML adapters under `backend/app/`. The frontend calls the backend API; Vite also proxies `/api` requests to `http://localhost:8000` during development. FastAPI's interactive API documentation is available at `/docs`.

## Models

The application supports multiple model architectures; the model catalog and inference status depend on which compatible checkpoint files are present and load successfully.

| Task | Model | Notes |
|---|---|---|
| MRI classification (default) | EfficientNetV2B0, 256 × 256 | CLAHE preprocessing; four candidate classes: glioma, meningioma, no tumor, and pituitary. |
| MRI classification | EfficientNetV2B0, legacy; Swin Transformer | Optional alternate checkpoints. |
| 2D lesion segmentation | BRISC binary U-Net; DeepLabV3+ | Binary lesion masks, not multi-region tumor subtyping. |
| 3D segmentation | BraTS multimodal 3D U-Net | Uses aligned FLAIR, T1, T1CE, and T2 volumes when its compatible checkpoint is available. |

Model files are loaded from `backend/model_artifacts/` (paths can be configured in the backend environment). Examples include `EfficientNetV2B0_clahe_256.keras`, `EfficientNetV2B0_clahe_iid_best.keras`, `BRISC_Binary_UNet_best.keras`, `deeplabv3plus.keras`, `swin_transformer.keras`, and `brats_3d_unet/best_3d_unet_model.keras`.

TensorFlow is not installed by `backend/requirements.txt`; install a TensorFlow/Keras version compatible with your Python and platform when running real model inference. In development, if a checkpoint is unavailable, the backend can return explicitly marked mock results (`is_mock: true`) when `USE_MOCK_MODELS_IN_DEV=true`. These are fixtures, not model predictions. In production, a missing model causes the relevant inference request to fail instead of substituting a mock.

The optional research Copilot is separate from the MRI models. It uses Groq when `GROQ_API_KEY` is configured, with `openai/gpt-oss-120b` as the default model and `openai/gpt-oss-20b` as the fallback. Without a key, live Groq chat is not configured.

## Run locally

### Prerequisites

- Python 3.11
- Node.js and npm
- MongoDB locally or a MongoDB Atlas connection string
- Optional: TensorFlow/Keras and model checkpoints for real inference; ImageKit and Groq credentials for those integrations

### 1. Configure and run the backend

In PowerShell, from the repository root:

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `backend/.env` and set `MONGODB_URL` to your local MongoDB or Atlas URI. The example uses `mongodb://localhost:27017`. Set a private random `SECRET_KEY` (for example, generate one with `python -c "import secrets; print(secrets.token_hex(32))"`); do not commit `.env` or production credentials. Configure ImageKit and Groq keys only if you need those integrations.

Start the API from the `backend` directory:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Check `http://localhost:8000/api/health` or open `http://localhost:8000/docs`.

### 2. Configure and run the frontend

Open a second PowerShell terminal at the repository root:

```powershell
cd frontend
Copy-Item .env.example .env
npm install
npm run dev
```

Open the URL Vite prints (by default, `http://localhost:5173`). The frontend environment example points to the local API at `http://localhost:8000`.

### Tests and production build

Backend tests (run from `backend` with its virtual environment activated):

```powershell
pytest
```

Frontend production build (run from `frontend`):

```powershell
npm run build
```

## Repository layout

```text
backend/     FastAPI application, ML services, and backend tests
frontend/    React + TypeScript web application
reference/   Reference model/preprocessing components and tests
docs/        Integration and validation notes
```
