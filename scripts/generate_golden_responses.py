import sys
import os
import json
import asyncio
from httpx import AsyncClient, ASGITransport

sys.path.insert(0, 'backend')
os.environ['PYTHONPATH'] = 'backend'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from app.main import app
from app.ml.model_loader import model_loader

GOLDEN_DIR = 'tests/golden'
os.makedirs(GOLDEN_DIR, exist_ok=True)

def sanitize_json(obj):
    """Normalize variable fields like UUIDs, timestamps, URLs with dynamic ids."""
    if isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            if k in ('analysis_id', 'prediction_id', 'id', 'user_id'):
                res[k] = "<DYNAMIC_ID>"
            elif k in ('createdAt', 'created_at', 'updated_at', 'analysisDate', 'inference_timestamp'):
                res[k] = "<TIMESTAMP>"
            elif k in ('processing_time_ms', 'inference_time_ms', 'inference_time'):
                res[k] = "<DURATION_MS>"
            elif isinstance(v, str) and (v.startswith('data:image') or 'ik.imagekit.io' in v or '/api/storage/' in v):
                # Keep presence flag
                res[k] = f"<IMAGE_DATA_URL len={len(v)}>"
            else:
                res[k] = sanitize_json(v)
        return res
    elif isinstance(obj, list):
        return [sanitize_json(i) for i in obj]
    return obj

async def main():
    print("Pre-loading models for golden capture...")
    model_loader.load_models()
    
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Models Catalog
        print("Capturing GET /api/classification/models...")
        r = await client.get("/api/classification/models")
        with open(os.path.join(GOLDEN_DIR, "classification_models.json"), "w") as f:
            json.dump(sanitize_json(r.json()), f, indent=2)
            
        print("Capturing GET /api/models...")
        r = await client.get("/api/models")
        with open(os.path.join(GOLDEN_DIR, "models_catalog.json"), "w") as f:
            json.dump(sanitize_json(r.json()), f, indent=2)

        # 2. 3D Segmentation Status
        print("Capturing GET /api/segmentation/status...")
        r = await client.get("/api/segmentation/status")
        with open(os.path.join(GOLDEN_DIR, "segmentation_status.json"), "w") as f:
            json.dump(sanitize_json(r.json()), f, indent=2)

        # 3. 2D Classification Analyze (EfficientNetV2B0 legacy default)
        sample_png = "backend/tests/samples/BraTS20_Training_001/png_previews/BraTS20_Training_001_flair_slice_67.png"
        if os.path.exists(sample_png):
            print("Capturing POST /api/classification/analyze (EfficientNetV2B0)...")
            with open(sample_png, "rb") as img_f:
                files = {"image": ("slice_67.png", img_f.read(), "image/png")}
                data = {"model_id": "efficientnetv2b0"}
                r = await client.post("/api/classification/analyze", files=files, data=data)
                with open(os.path.join(GOLDEN_DIR, "classification_analyze_efficientnetv2b0.json"), "w") as f:
                    json.dump(sanitize_json(r.json()), f, indent=2)

            print("Capturing POST /api/classification/analyze (Swin Transformer)...")
            with open(sample_png, "rb") as img_f:
                files = {"image": ("slice_67.png", img_f.read(), "image/png")}
                data = {"model_id": "swin_transformer"}
                r = await client.post("/api/classification/analyze", files=files, data=data)
                with open(os.path.join(GOLDEN_DIR, "classification_analyze_swin.json"), "w") as f:
                    json.dump(sanitize_json(r.json()), f, indent=2)

        # 4. 3D Validate Study
        sample_dir = "backend/tests/samples/BraTS20_Training_001"
        if os.path.exists(os.path.join(sample_dir, "BraTS20_Training_001_flair.nii")):
            print("Capturing POST /api/segmentation/validate-study...")
            files = {
                "flair": ("flair.nii", open(os.path.join(sample_dir, "BraTS20_Training_001_flair.nii"), "rb").read(), "application/octet-stream"),
                "t1": ("t1.nii", open(os.path.join(sample_dir, "BraTS20_Training_001_t1.nii"), "rb").read(), "application/octet-stream"),
                "t1ce": ("t1ce.nii", open(os.path.join(sample_dir, "BraTS20_Training_001_t1ce.nii"), "rb").read(), "application/octet-stream"),
                "t2": ("t2.nii", open(os.path.join(sample_dir, "BraTS20_Training_001_t2.nii"), "rb").read(), "application/octet-stream"),
                "seg": ("seg.nii", open(os.path.join(sample_dir, "BraTS20_Training_001_seg.nii"), "rb").read(), "application/octet-stream")
            }
            r = await client.post("/api/segmentation/validate-study", files=files)
            with open(os.path.join(GOLDEN_DIR, "segmentation_validate_study.json"), "w") as f:
                json.dump(sanitize_json(r.json()), f, indent=2)

    print("Golden responses captured successfully in tests/golden/:")
    for f in os.listdir(GOLDEN_DIR):
        print("  -", f)

if __name__ == "__main__":
    asyncio.run(main())

