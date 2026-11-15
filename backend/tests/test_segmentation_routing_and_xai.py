import pytest
import io
import numpy as np
from PIL import Image
import nibabel as nib
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.models.user import UserDB
from app.ml.model_loader import model_loader
from app.api.deps import get_current_user_or_default

model_loader.load_models()


def create_dummy_png():
    img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    image = Image.fromarray(img_array)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def create_dummy_nifti():
    arr = np.random.rand(16, 16, 16).astype(np.float32)
    img = nib.Nifti1Image(arr, np.eye(4))
    buf = io.BytesIO()
    file_map = img.make_file_map({'image': buf, 'header': buf})
    img.to_file_map(file_map)
    return buf.getvalue()


@pytest.fixture
def override_auth():
    test_user = UserDB(
        id="test_user_id_xai",
        email="doctor@example.com",
        name="Dr. Neurologist",
        role="clinician",
        is_verified=True,
        password_hash="testhash123",
    )
    app.dependency_overrides[get_current_user_or_default] = lambda: test_user
    yield test_user
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_classification_and_segmentation_workflow_type_isolation(override_auth):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        png_bytes = create_dummy_png()

        # 1. EfficientNetV2B0 2D Classification
        resp_eff = await ac.post(
            "/api/classification/analyze",
            files={"image": ("brain_scan.png", png_bytes, "image/png")},
            data={"model_id": "efficientnetv2b0"},
        )
        assert resp_eff.status_code == 200, resp_eff.text
        data_eff = resp_eff.json()
        assert data_eff["workflow_type"] == "2D_CLASSIFICATION"
        assert data_eff["actual_model"]["model_id"] == "efficientnetv2b0"
        assert data_eff.get("gradcam_overlay") is not None
        assert data_eff.get("gradcam_pp_overlay") is not None
        eff_analysis_id = data_eff["analysis_id"]

        # Fetch full analysis record
        get_eff = await ac.get(f"/api/analyses/{eff_analysis_id}")
        assert get_eff.status_code == 200
        eff_full = get_eff.json()
        assert eff_full["workflow_type"] == "2D_CLASSIFICATION"
        assert eff_full["analysis_mode"] == "classification_2d"
        assert eff_full["actual_model"]["model_id"] == "efficientnetv2b0"
        assert eff_full["gradcam_overlay"] is not None
        assert eff_full["gradcam_pp_overlay"] is not None

        # 2. Swin Transformer 2D Classification
        resp_swin = await ac.post(
            "/api/classification/analyze",
            files={"image": ("brain_swin.png", png_bytes, "image/png")},
            data={"model_id": "swin_transformer"},
        )
        assert resp_swin.status_code == 200, resp_swin.text
        data_swin = resp_swin.json()
        assert data_swin["workflow_type"] == "2D_CLASSIFICATION"
        assert data_swin["actual_model"]["model_id"] == "swin_transformer"
        assert data_swin["actual_model"]["parameters_count"] == "28.3M params"
        assert data_swin.get("gradcam_overlay") is not None
        swin_analysis_id = data_swin["analysis_id"]

        # 3. 3D Multimodal Segmentation
        nii_bytes = create_dummy_nifti()
        files = {
            "flair": ("bra_flair.nii", nii_bytes, "application/octet-stream"),
            "t1": ("bra_t1.nii", nii_bytes, "application/octet-stream"),
            "t1ce": ("bra_t1ce.nii", nii_bytes, "application/octet-stream"),
            "t2": ("bra_t2.nii", nii_bytes, "application/octet-stream"),
        }
        resp_seg = await ac.post(
            "/api/segmentation/analyze",
            files=files,
            data={"model_id": "brats_3d_unet"},
        )
        assert resp_seg.status_code == 200, resp_seg.text
        data_seg = resp_seg.json()
        assert data_seg["workflow_type"] == "3D_SEGMENTATION"
        assert data_seg["analysis_mode"] == "segmentation_3d"
        assert data_seg["actual_model"]["model_id"] == "brats_3d_unet"
        assert data_seg["actual_model"]["task_type"] == "3D Multimodal Volumetric Segmentation"
        seg_analysis_id = data_seg["analysis_id"]

        # Fetch 3D analysis record
        get_seg = await ac.get(f"/api/analyses/{seg_analysis_id}")
        assert get_seg.status_code == 200
        seg_full = get_seg.json()
        assert seg_full["workflow_type"] == "3D_SEGMENTATION"
        assert seg_full["analysis_mode"] == "segmentation_3d"
        assert seg_full["actual_model"]["model_id"] == "brats_3d_unet"
        assert "slices" in seg_full
        assert "surface_3d" in seg_full


@pytest.mark.asyncio
async def test_xai_model_provenance_and_gradcam_pp_behavior(override_auth):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        png_bytes = create_dummy_png()

        # Create a Swin Transformer analysis
        resp_swin = await ac.post(
            "/api/classification/analyze",
            files={"image": ("swin_sample.png", png_bytes, "image/png")},
            data={"model_id": "swin_transformer"},
        )
        assert resp_swin.status_code == 200
        swin_id = resp_swin.json()["analysis_id"]

        # 1. Swin Transformer on-demand Grad-CAM: should resolve to swin layer and succeed
        explain_gradcam = await ac.post(
            f"/api/analyses/{swin_id}/explain",
            json={"method": "gradcam"}
        )
        assert explain_gradcam.status_code == 200, explain_gradcam.text
        gradcam_res = explain_gradcam.json()
        assert gradcam_res["method"] == "gradcam"
        assert "swin_transformer_block_7" in gradcam_res["target_layer"]
        assert gradcam_res["overlay_url"] is not None

        # 2. Swin Transformer on-demand Grad-CAM++: Swin ViT does not support CNN Grad-CAM++, should return 400 or 422
        explain_pp = await ac.post(
            f"/api/analyses/{swin_id}/explain",
            json={"method": "gradcam_pp"}
        )
        assert explain_pp.status_code in (400, 422)
        assert "Swin Transformer supports validated transformer Grad-CAM only" in explain_pp.json()["detail"]

        # 3. EfficientNetV2B0 on-demand Grad-CAM++: should succeed and resolve to top_conv
        resp_eff = await ac.post(
            "/api/classification/analyze",
            files={"image": ("eff_sample.png", png_bytes, "image/png")},
            data={"model_id": "efficientnetv2b0"},
        )
        assert resp_eff.status_code == 200
        eff_id = resp_eff.json()["analysis_id"]

        eff_pp = await ac.post(
            f"/api/analyses/{eff_id}/explain",
            json={"method": "gradcam_pp"}
        )
        assert eff_pp.status_code == 200, eff_pp.text
        eff_pp_res = eff_pp.json()
        assert eff_pp_res["method"] == "gradcam_pp"
        assert eff_pp_res["target_layer"] == "top_conv"
        assert eff_pp_res["overlay_url"] is not None
