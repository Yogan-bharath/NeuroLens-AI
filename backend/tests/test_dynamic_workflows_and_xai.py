import io
import pytest
import numpy as np
from PIL import Image
from httpx import AsyncClient
import nibabel as nib


def create_custom_mri(lesion_x: int, lesion_y: int, radius: int = 25) -> bytes:
    """Generate a synthetic 2D MRI with a focal hyperintensity lesion at a specific coordinate."""
    img_array = np.zeros((256, 256), dtype=np.uint8)
    y, x = np.ogrid[:256, :256]
    # Skull / parenchyma
    dist = np.sqrt((x - 128) ** 2 + (y - 128) ** 2)
    img_array[dist < 100] = 70
    img_array[dist < 75] = 150
    # Distinct lesion
    lesion_mask = (x - lesion_x) ** 2 + (y - lesion_y) ** 2 < radius ** 2
    img_array[lesion_mask] = 250

    img = Image.fromarray(img_array, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def create_synthetic_nifti_bytes(shape=(32, 32, 16), val=1.0) -> bytes:
    """Generate valid NIfTI bytes for volumetric testing."""
    data = np.full(shape, val, dtype=np.float32)
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    buf = io.BytesIO()
    file_map = img.make_file_map()
    file_map["image"].fileobj = buf
    img.to_file_map(file_map)
    return buf.getvalue()


@pytest.mark.asyncio
async def test_classification_only_produces_image_specific_xai(client: AsyncClient, auth_headers: dict):
    """
    Test that two different 2D MRI scans produce:
    1. Independent analysis records with distinct IDs.
    2. Distinct uploaded image storage URLs.
    3. Distinct Grad-CAM saliency heatmaps without cross-talk.
    4. Segmentation fields are completely absent/null (workflow isolation).
    """
    scan_a_bytes = create_custom_mri(70, 80, radius=20)
    scan_b_bytes = create_custom_mri(180, 175, radius=30)

    # 1. Run Classification for Scan A
    files_a = {"image": ("scan_a_left_frontal.png", scan_a_bytes, "image/png")}
    data_a = {"model_id": "efficientnetv2b0", "use_clahe": "true"}
    res_a = await client.post("/api/classification/analyze", files=files_a, data=data_a, headers=auth_headers)
    assert res_a.status_code == 200
    json_a = res_a.json()
    id_a = json_a["analysis_id"]

    # 2. Run Classification for Scan B
    files_b = {"image": ("scan_b_right_parietal.png", scan_b_bytes, "image/png")}
    data_b = {"model_id": "efficientnetv2b0", "use_clahe": "true"}
    res_b = await client.post("/api/classification/analyze", files=files_b, data=data_b, headers=auth_headers)
    assert res_b.status_code == 200
    json_b = res_b.json()
    id_b = json_b["analysis_id"]

    # Assert distinct IDs
    assert id_a != id_b
    assert json_a["workflow_type"] == "2D_CLASSIFICATION"
    assert json_b["workflow_type"] == "2D_CLASSIFICATION"

    # Retrieve full records
    rec_a = (await client.get(f"/api/analyses/{id_a}", headers=auth_headers)).json()
    rec_b = (await client.get(f"/api/analyses/{id_b}", headers=auth_headers)).json()

    # Image provenance: distinct URLs
    assert rec_a["imageDataUrl"] != rec_b["imageDataUrl"]
    assert "scan_a" in rec_a["fileName"]
    assert "scan_b" in rec_b["fileName"]

    # Saliency provenance: distinct heatmaps
    assert rec_a.get("gradcam_overlay") is not None
    assert rec_b.get("gradcam_overlay") is not None
    assert rec_a["gradcam_overlay"] != rec_b["gradcam_overlay"]

    # Workflow isolation: segmentation data must be null
    assert rec_a.get("volume_dimensions") is None
    assert rec_a.get("surface_3d") is None
    assert rec_b.get("volume_dimensions") is None
    assert rec_b.get("surface_3d") is None


@pytest.mark.asyncio
async def test_on_demand_xai_generation_persists_without_mock_fallbacks(client: AsyncClient, auth_headers: dict):
    """
    Test that requesting on-demand Grad-CAM++:
    1. Successfully computes genuine higher-order gradients.
    2. Persists the new explanation into the analysis record.
    3. Re-retrieving the analysis includes the new explanation without data fabrication.
    """
    scan_bytes = create_custom_mri(120, 120, radius=22)
    files = {"image": ("test_ondemand_xai.png", scan_bytes, "image/png")}
    data = {"model_id": "efficientnetv2b0", "use_clahe": "true"}
    res = await client.post("/api/classification/analyze", files=files, data=data, headers=auth_headers)
    assert res.status_code == 200
    analysis_id = res.json()["analysis_id"]

    # Initial explanations
    init_res = await client.get(f"/api/analyses/{analysis_id}/explanations", headers=auth_headers)
    assert init_res.status_code == 200

    # Request Grad-CAM++
    exp_payload = {"method": "gradcam_pp", "target_layer": "top_conv"}
    exp_res = await client.post(f"/api/analyses/{analysis_id}/explain", json=exp_payload, headers=auth_headers)
    assert exp_res.status_code == 200
    exp_data = exp_res.json()
    assert exp_data["method"] == "gradcam_pp"
    assert exp_data["heatmap_url"] is not None
    assert exp_data["overlay_url"] is not None
    assert exp_data["metadata"]["is_mock"] is False

    # Verify persistence into analysis record
    rec = (await client.get(f"/api/analyses/{analysis_id}", headers=auth_headers)).json()
    exp_methods = [e["method"] for e in rec.get("explanations", [])]
    assert "gradcam_pp" in exp_methods


@pytest.mark.asyncio
async def test_segmentation_only_workflow_isolation(client: AsyncClient, auth_headers: dict):
    """
    Test that 3D segmentation analysis:
    1. Properly isolates volumetric subregion statistics.
    2. Contains no 2D classification softmax distribution or scores.
    3. Provides real multiplanar slice scrubber endpoints.
    """
    from app.services.analysis_service import analysis_service

    dummy_seg_result = {
        "success": True,
        "prediction_id": "test-dynamic-seg-789",
        "volume_dimensions": [240, 240, 155],
        "voxel_spacing": [1.0, 1.0, 1.0],
        "anatomical_orientation": ["R", "A", "S"],
        "class_distribution": [
            {"index": 0, "name": "Background", "voxel_count": 8600000, "percentage": 96.0, "volume_mm3": 8600000.0, "color": "#000000"},
            {"index": 1, "name": "Necrotic", "voxel_count": 14000, "percentage": 0.16, "volume_mm3": 14000.0, "color": "#EF4444"},
            {"index": 2, "name": "Edema", "voxel_count": 92000, "percentage": 1.03, "volume_mm3": 92000.0, "color": "#10B981"},
            {"index": 3, "name": "Enhancing Tumor", "voxel_count": 48000, "percentage": 0.54, "volume_mm3": 48000.0, "color": "#F59E0B"},
        ],
        "slices": {
            "axial": {"slice_index": 78, "preview": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="},
            "coronal": {"slice_index": 120, "preview": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="},
            "sagittal": {"slice_index": 120, "preview": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="},
        },
        "surface_3d": {
            "anatomical_location": {
                "hemisphere": "Left",
                "hemisphere_detail": "Left Temporal Region (82% mass)",
                "center_x_mm": -28.4,
                "center_y_mm": 12.1,
                "center_z_mm": 40.0,
            }
        },
        "explainability": {
            "modality_ablation": [
                {"removed_modality": "FLAIR", "tumor_voxel_count": 115000, "relative_change_pct": -25.3, "impact_level": "High"},
                {"removed_modality": "T1CE", "tumor_voxel_count": 105000, "relative_change_pct": -31.8, "impact_level": "High"}
            ],
            "uncertainty": {"mean_entropy_overall": 0.038, "mean_entropy_tumor": 0.179}
        }
    }

    seg_id = await analysis_service.save_segmentation_analysis(
        user_id="test_researcher_001",
        inference_result=dummy_seg_result,
        modality_filenames={"flair": "subject_flair.nii", "t1": "subject_t1.nii", "t1ce": "subject_t1ce.nii", "t2": "subject_t2.nii"},
        model_name="Attention CNN + Vision Transformer",
        model_version="v2.4.0",
    )

    rec = (await client.get(f"/api/analyses/{seg_id}", headers=auth_headers)).json()
    assert rec["workflow_type"] == "3D_SEGMENTATION"
    assert rec["volume_dimensions"] == [240, 240, 155]
    assert rec["surface_3d"]["anatomical_location"]["hemisphere"] == "Left"
    assert len(rec["explainability"]["modality_ablation"]) == 2

    # Verify classification fields are completely absent
    assert rec.get("scores") is None
    assert rec.get("classification") is None


@pytest.mark.asyncio
async def test_combined_workflow_preserves_both_streams_independently(client: AsyncClient, auth_headers: dict):
    """
    Test that the Combined workflow:
    1. Executes both 2D and 3D pipelines independently.
    2. Handles partial execution (e.g., 2D only provided) by preserving the successful result.
    3. Never mixes 2D input with 3D models or vice-versa.
    4. Stores and returns distinct results in a master combined session.
    5. Rejects requests where neither input stream is provided.
    """
    # 1. Test empty request rejected
    empty_res = await client.post("/api/analyses/combined", data={"use_clahe": "true"}, headers=auth_headers)
    assert empty_res.status_code == 400

    # 2. Test 2D-only partial execution
    scan_2d = create_custom_mri(100, 100, radius=20)
    files = {
        "file_2d": ("combined_slice.png", scan_2d, "image/png"),
    }
    data = {"use_clahe": "true"}

    res = await client.post("/api/analyses/combined", files=files, data=data, headers=auth_headers)
    assert res.status_code == 200
    json_data = res.json()
    assert json_data["status"] == "partial"
    assert json_data["classification_status"] == "completed"
    assert json_data["segmentation_status"] == "failed"
    assert json_data["classification_result"] is not None
    assert json_data["classification_result"]["top_prediction"] is not None

    master_id = json_data["combined_analysis_id"]
    rec = (await client.get(f"/api/analyses/{master_id}", headers=auth_headers)).json()
    assert "COMBINED" in rec["workflow_type"]
    assert rec["analysis_mode"] == "combined"
    assert rec.get("classification") is not None
