import io
import pytest
from httpx import AsyncClient, ASGITransport
import numpy as np
import nibabel as nib
from PIL import Image

from app.main import app
from app.ml.model_loader import model_loader


def _create_sample_png() -> bytes:
    img = Image.fromarray(np.random.randint(40, 220, (128, 128), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _create_sample_nifti() -> bytes:
    affine = np.eye(4)
    data = np.zeros((16, 16, 16), dtype=np.float32)
    data[4:12, 4:12, 4:12] = 100.0
    img = nib.Nifti1Image(data, affine)
    buf = io.BytesIO()
    file_map = img.make_file_map({'image': buf, 'header': buf})
    img.to_file_map(file_map)
    return buf.getvalue()


# Ensure models are loaded
model_loader.load_models()


@pytest.mark.asyncio
async def test_classification_model_selection_persistence_and_provenance():
    """Verify 2D classification preserves selected model metadata from inference to storage to retrieval."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        image_bytes = _create_sample_png()
        files = {"image": ("test_brain.png", image_bytes, "image/png")}
        data = {"model_id": "efficientnetv2b0", "use_clahe": "true"}

        # 1. Inference Request
        resp = await client.post("/api/classification/analyze", files=files, data=data)
        assert resp.status_code == 200, resp.text
        json_resp = resp.json()

        assert json_resp["success"] is True
        assert json_resp["model_id"] == "efficientnetv2b0"
        assert json_resp["model_name"] == "EfficientNetV2B0"
        assert "actual_model" in json_resp
        actual_model = json_resp["actual_model"]
        assert actual_model["model_id"] == "efficientnetv2b0"
        assert actual_model["model_name"] == "EfficientNetV2B0"
        assert "Fused-MBConv" in actual_model["architecture"]
        assert "128" in actual_model["input_resolution"]
        assert actual_model["task_type"] == "2D Brain Tumor Classification"

        analysis_id = json_resp["analysis_id"]

        # 2. Retrieve Stored Analysis
        get_resp = await client.get(f"/api/analyses/{analysis_id}")
        assert get_resp.status_code == 200, get_resp.text
        stored = get_resp.json()

        assert stored["id"] == analysis_id
        assert stored["model_id"] == "efficientnetv2b0"
        assert stored["model_name"] == "EfficientNetV2B0"
        assert stored["actual_model"] is not None
        assert stored["actual_model"]["model_id"] == "efficientnetv2b0"
        assert stored["actual_model"]["model_name"] == "EfficientNetV2B0"
        assert stored["actual_model"]["architecture"] == actual_model["architecture"]


@pytest.mark.asyncio
async def test_classification_rejects_unavailable_model_without_silent_fallback():
    """Verify backend rejects uninstalled or invalid models explicitly without silent fallbacks."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        image_bytes = _create_sample_png()
        files = {"image": ("test_brain.png", image_bytes, "image/png")}
        data = {"model_id": "non_existent_fake_resnet99"}

        resp = await client.post("/api/classification/analyze", files=files, data=data)
        assert resp.status_code == 400
        assert "not installed" in resp.json()["detail"].lower() or "not available" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_segmentation_model_selection_persistence_and_provenance():
    """Verify 3D segmentation preserves selected model metadata from inference to storage to retrieval."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create minimal valid NIfTI files
        affine = np.eye(4)
        vols = {}
        for m in ["flair", "t1", "t1ce", "t2"]:
            data = np.zeros((16, 16, 16), dtype=np.float32)
            data[4:12, 4:12, 4:12] = 50.0
            img = nib.Nifti1Image(data, affine)
            buf = io.BytesIO()
            file_map = img.make_file_map({'image': buf, 'header': buf})
            img.to_file_map(file_map)
            vols[m] = buf.getvalue()

        files = [
            ("flair", ("study_flair.nii", vols["flair"], "application/octet-stream")),
            ("t1", ("study_t1.nii", vols["t1"], "application/octet-stream")),
            ("t1ce", ("study_t1ce.nii", vols["t1ce"], "application/octet-stream")),
            ("t2", ("study_t2.nii", vols["t2"], "application/octet-stream")),
        ]
        data = {"model_id": "brats_3d_unet"}

        resp = await client.post("/api/segmentation/analyze", files=files, data=data)
        assert resp.status_code == 200, resp.text
        json_resp = resp.json()

        assert json_resp["success"] is True
        assert json_resp["model_id"] == "brats_3d_unet"
        assert "actual_model" in json_resp
        actual_model = json_resp["actual_model"]
        assert actual_model["model_id"] == "brats_3d_unet"
        assert actual_model["task_type"] == "3D Multimodal Volumetric Segmentation"
        assert actual_model["input_modalities"] == ["FLAIR", "T1", "T1CE", "T2"]

        analysis_id = json_resp["analysis_id"]

        # Retrieve stored analysis
        get_resp = await client.get(f"/api/analyses/{analysis_id}")
        assert get_resp.status_code == 200, get_resp.text
        stored = get_resp.json()

        assert stored["id"] == analysis_id
        assert stored["model_id"] == "brats_3d_unet"
        assert stored["actual_model"] is not None
        assert stored["actual_model"]["model_id"] == "brats_3d_unet"
        assert stored["actual_model"]["task_type"] == "3D Multimodal Volumetric Segmentation"


@pytest.mark.asyncio
async def test_history_preserves_immutable_model_identity():
    """Verify history listing returns the historical model metadata for each analysis session."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create an analysis session
        image_bytes = _create_sample_png()
        files = {"image": ("test_brain.png", image_bytes, "image/png")}
        data = {"model_id": "efficientnetv2b0"}
        resp = await client.post("/api/classification/analyze", files=files, data=data)
        assert resp.status_code == 200
        analysis_id = resp.json()["analysis_id"]

        # Fetch history
        hist_resp = await client.get("/api/analyses")
        assert hist_resp.status_code == 200
        items = hist_resp.json()
        target = next((item for item in items if item["id"] == analysis_id), None)

        assert target is not None
        assert target["model_id"] == "efficientnetv2b0"
        assert target["model_name"] == "EfficientNetV2B0"
        assert target["actual_model"] is not None
        assert target["actual_model"]["model_id"] == "efficientnetv2b0"
