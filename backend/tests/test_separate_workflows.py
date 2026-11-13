import io
import os
import sys
import numpy as np
import nibabel as nib
from PIL import Image
import pytest
from fastapi.testclient import TestClient

# Ensure backend root is on path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from app.main import app
from app.ml.brats_3d_unet import (
    build_3d_unet,
    PATCH_SIZE,
    N_CHANNELS,
    NUM_CLASSES,
    GRAD_CAM_LAYER,
    make_starts,
    dice_iou_binary,
    evaluate_volume,
    prediction_entropy,
    ablate_modality,
    extract_tumor_center_patch,
    make_3d_gradcam,
)
from app.services.brats_3d_service import (
    normalize_modality,
    prepare_multimodal_volumes,
    brats_3d_segmentation_service,
)
from app.ml.model_loader import model_loader

# Ensure models are loaded for testing
model_loader.load_models()
client = TestClient(app)


def test_workflow_a_classification_models_catalog():
    """Verify classification models catalog returns available models and flags."""
    resp = client.get("/api/classification/models")
    assert resp.status_code == 200
    models = resp.json()
    model_ids = [m["id"] for m in models]
    assert "efficientnetv2b0" in model_ids
    assert "densenet121" in model_ids
    assert "resnet50" in model_ids

    # Verify models without checkpoints are marked coming_soon
    for m in models:
        if m["id"] == "densenet121":
            assert m["status"] == "coming_soon"


def test_workflow_a_classification_valid_png():
    """Test 2D classification with a valid synthetic PNG image."""
    # Create synthetic 2D MRI slice (128x128 grayscale PNG)
    arr = np.random.randint(20, 220, (128, 128), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    resp = client.post(
        "/api/classification/analyze",
        files={"image": ("sample_mri.png", buf, "image/png")},
        data={"model_id": "efficientnetv2b0", "use_clahe": "false"},
    )
    print("CLASSIFICATION RESPONSE:", resp.status_code, resp.json())
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "prediction_id" in data
    assert "top_prediction" in data
    assert "confidence" in data
    assert isinstance(data["scores"], list)
    assert len(data["scores"]) == 4
    assert "disclaimer" in data


def test_workflow_a_classification_rejects_nifti():
    """Ensure NIfTI files are rejected by the 2D classification pipeline."""
    dummy_nii = b"dummy_nifti_data"
    resp = client.post(
        "/api/classification/analyze",
        files={"image": ("brain.nii", dummy_nii, "application/octet-stream")},
        data={"model_id": "efficientnetv2b0"},
    )
    assert resp.status_code == 400
    assert "NIfTI" in resp.json()["detail"]


def test_workflow_b_segmentation_status():
    """Verify 3D segmentation status endpoint."""
    resp = client.get("/api/segmentation/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "checkpoint_exists" in data
    assert "architecture_initialized" in data
    assert data["architecture_initialized"] is True
    assert "tensorflow_version" in data
    assert "model_details" in data
    assert data["model_details"]["patch_size"] == [96, 96, 64]
    assert data["model_details"]["target_xai_layer"] == "bottleneck_relu2"


def test_workflow_b_segmentation_rejects_png():
    """Ensure standard 2D PNG/JPG images are rejected in 3D segmentation."""
    dummy_png = b"\x89PNG\r\n\x1a\n"
    resp = client.post(
        "/api/segmentation/analyze",
        files={
            "flair": ("brain.png", dummy_png, "image/png"),
            "t1": ("brain.png", dummy_png, "image/png"),
            "t1ce": ("brain.png", dummy_png, "image/png"),
            "t2": ("brain.png", dummy_png, "image/png"),
        },
    )
    assert resp.status_code == 400
    assert "must be a NIfTI format" in resp.json()["detail"]


def test_workflow_b_segmentation_rejects_missing_modality():
    """Ensure missing required modalities are rejected by FastAPI."""
    dummy_nii = b"dummy"
    resp = client.post(
        "/api/segmentation/analyze",
        files={
            "flair": ("flair.nii", dummy_nii, "application/octet-stream"),
            "t1": ("t1.nii", dummy_nii, "application/octet-stream"),
            # missing t1ce and t2
        },
    )
    assert resp.status_code == 422  # Validation error for missing required files


def test_exact_notebook_3d_unet_architecture():
    """Verify the 3D U-Net matches braingliomav6 (2).ipynb."""
    model = build_3d_unet()
    assert model.input_shape == (None, 96, 96, 64, 4)
    assert model.output_shape == (None, 96, 96, 64, 4)
    assert model.count_params() == 5606628
    layer_names = [l.name for l in model.layers]
    assert GRAD_CAM_LAYER in layer_names
    assert "segmentation_output" in layer_names


def test_exact_notebook_normalization():
    """Verify non-zero voxel normalization matching Cell 12."""
    vol = np.zeros((10, 10, 10), dtype=np.float32)
    vol[2:8, 2:8, 2:8] = np.random.uniform(50, 150, (6, 6, 6))

    norm = normalize_modality(vol)
    # Background must remain exactly zero
    assert np.all(norm[vol == 0] == 0.0)
    # Non-zero region must have mean ~0 and std ~1
    nz_vals = norm[vol != 0]
    np.testing.assert_allclose(nz_vals.mean(), 0.0, atol=1e-5)
    np.testing.assert_allclose(nz_vals.std(), 1.0, atol=1e-5)


def test_exact_notebook_evaluation_metrics():
    """Verify evaluation metrics matching Cell 36."""
    gt = np.zeros((20, 20, 20), dtype=np.uint8)
    pred = np.zeros((20, 20, 20), dtype=np.uint8)

    # Class 1: Necrotic, Class 2: Edema, Class 3: Enhancing
    gt[2:6, 2:6, 2:6] = 1
    pred[2:6, 2:6, 2:6] = 1

    gt[7:12, 7:12, 7:12] = 2
    pred[7:12, 7:12, 7:12] = 2

    gt[13:17, 13:17, 13:17] = 3
    pred[13:17, 13:17, 13:17] = 3

    metrics = evaluate_volume(gt, pred)
    assert metrics["whole_tumor_dice"] == 1.0
    assert metrics["whole_tumor_iou"] == 1.0
    assert metrics["tumor_core_dice"] == 1.0
    assert metrics["necrotic_dice"] == 1.0
    assert metrics["edema_dice"] == 1.0
    assert metrics["enhancing_dice"] == 1.0
    assert metrics["pixel_accuracy"] == 1.0


def test_exact_notebook_sliding_window_starts():
    """Verify make_starts() stride calculations."""
    starts = make_starts(240, 96, 64)
    assert starts[0] == 0
    assert starts[-1] == 240 - 96  # 144
    assert len(starts) == 4  # [0, 64, 128, 144]


def test_exact_notebook_predictive_entropy():
    """Verify normalized entropy matching Cell 56."""
    probs = np.full((5, 5, 5, 4), 0.25, dtype=np.float32)
    entropy = prediction_entropy(probs)
    # Uniform distribution has maximum entropy (1.0)
    np.testing.assert_allclose(entropy, 1.0, atol=1e-4)

    # Completely certain distribution has minimum entropy (0.0)
    certain_probs = np.zeros((5, 5, 5, 4), dtype=np.float32)
    certain_probs[..., 0] = 1.0
    certain_entropy = prediction_entropy(certain_probs)
    np.testing.assert_allclose(certain_entropy, 0.0, atol=1e-4)
