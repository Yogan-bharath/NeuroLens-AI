import pytest
from httpx import AsyncClient
from app.services.analysis_service import analysis_service
from app.services.metrics_service import metrics_service


@pytest.mark.asyncio
async def test_segmentation_persistence_and_retrieval(client: AsyncClient, auth_headers: dict):
    # Simulate a multimodal 3D segmentation inference result (e.g. from Attention CNN + ViT)
    dummy_result = {
        "success": True,
        "prediction_id": "test-pred-attvit-123",
        "download_url": "/api/segmentation/download/test-pred-attvit-123",
        "volume_dimensions": [240, 240, 155],
        "voxel_spacing": [1.0, 1.0, 1.0],
        "anatomical_orientation": ["R", "A", "S"],
        "processing_time_ms": 182,
        "class_distribution": [
            {"index": 0, "name": "Background", "voxel_count": 8500000, "percentage": 95.2, "volume_mm3": 8500000.0, "color": "#000000"},
            {"index": 1, "name": "Necrotic", "voxel_count": 12000, "percentage": 0.13, "volume_mm3": 12000.0, "color": "#EF4444"},
            {"index": 2, "name": "Edema", "voxel_count": 85000, "percentage": 0.95, "volume_mm3": 85000.0, "color": "#10B981"},
            {"index": 3, "name": "Enhancing Tumor", "voxel_count": 45000, "percentage": 0.50, "volume_mm3": 45000.0, "color": "#F59E0B"},
        ],
        "slices": {
            "axial": {"slice_index": 78, "preview": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="},
            "coronal": {"slice_index": 120, "preview": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="},
            "sagittal": {"slice_index": 120, "preview": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="},
        },
        "surface_3d": {
            "bounding_box": {"min": [-0.5, -0.5, -0.5], "max": [0.5, 0.5, 0.5], "center": [0.0, 0.0, 0.0]},
            "classes": [{"class_idx": 3, "name": "Enhancing Tumor", "hex": "#F59E0B"}],
            "points": {"3": [[0.1, 0.2, 0.3]]},
            "brain_surface": [[-0.4, 0.1, 0.2]],
            "anatomical_location": {
                "hemisphere": "Right",
                "hemisphere_detail": "Right Hemisphere (78.5% tumor mass)",
                "anterior_posterior": "Front (Anterior)",
                "ap_detail": "Frontal region (64% anterior)",
                "superior_inferior": "Upper (Superior)",
                "si_detail": "High vertex (65% height)",
                "summary": "Right Hemisphere · Front (Anterior) · Upper (Superior)",
                "center_x_mm": 24.5,
                "center_y_mm": 18.2,
                "center_z_mm": 35.0,
            },
        },
        "evaluation_metrics": {
            "whole_tumor_dice": 0.8937,
            "whole_tumor_iou": 0.8078,
            "enhancing_tumor_dice": 0.8123,
            "tumor_core_dice": 0.7716,
        },
        "explainability": {
            "gradcam_layer": "MultiHeadAttention",
            "modality_ablation": [{"removed_modality": "FLAIR", "tumor_voxel_count": 110000, "relative_change_pct": -22.5, "impact_level": "Moderate"}],
            "uncertainty": {"mean_entropy_overall": 0.0412, "mean_entropy_tumor": 0.1845},
        },
        "disclaimer": "Research model prediction only.",
    }

    modality_filenames = {
        "flair": "BraTS20_Training_001_flair.nii",
        "t1": "BraTS20_Training_001_t1.nii",
        "t1ce": "BraTS20_Training_001_t1ce.nii",
        "t2": "BraTS20_Training_001_t2.nii",
    }

    # 1. Save Attention CNN + ViT 3D segmentation
    analysis_id = await analysis_service.save_segmentation_analysis(
        user_id="test_researcher_001",
        inference_result=dummy_result,
        modality_filenames=modality_filenames,
        model_name="Attention CNN + Vision Transformer",
        model_version="v2.4.0",
    )
    assert analysis_id.startswith("ANL-")

    # 2. Retrieve analysis by ID
    analysis = await analysis_service.get_analysis(
        analysis_id=analysis_id,
        user_id="test_researcher_001",
        is_admin=False,
    )
    assert analysis is not None
    assert analysis.id == analysis_id
    assert analysis.analysis_mode == "segmentation_3d"
    assert analysis.volume_dimensions == [240, 240, 155]
    assert analysis.download_url == "/api/segmentation/download/test-pred-attvit-123"
    assert analysis.evaluation_metrics["whole_tumor_dice"] == 0.8937
    assert analysis.surface_3d["anatomical_location"]["hemisphere"] == "Right"

    # 3. Retrieve through HTTP API GET /api/analyses/{id}
    res = await client.get(f"/api/analyses/{analysis_id}", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["analysis_id"] == analysis_id
    assert data["volume_dimensions"] == [240, 240, 155]
    assert data["surface_3d"]["anatomical_location"]["summary"] == "Right Hemisphere · Front (Anterior) · Upper (Superior)"

    # 4. List through HTTP API GET /api/analyses
    list_res = await client.get("/api/analyses", headers=auth_headers)
    assert list_res.status_code == 200
    list_items = list_res.json()
    assert any(item["id"] == analysis_id for item in list_items)


@pytest.mark.asyncio
async def test_classification_persistence_and_retrieval(client: AsyncClient, auth_headers: dict):
    dummy_cls = {
        "prediction_id": "test-pred-cls-456",
        "top_prediction": "Glioma",
        "confidence": 0.942,
        "scores": [
            {"label": "Glioma", "score": 0.942, "percentage": 94.2, "isTopPrediction": True},
            {"label": "Meningioma", "score": 0.038, "percentage": 3.8, "isTopPrediction": False},
            {"label": "Pituitary", "score": 0.012, "percentage": 1.2, "isTopPrediction": False},
            {"label": "No Tumor", "score": 0.008, "percentage": 0.8, "isTopPrediction": False},
        ],
        "gradcam_overlay": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        "processing_time_ms": 45,
    }

    analysis_id = await analysis_service.save_classification_analysis(
        user_id="test_researcher_001",
        inference_result=dummy_cls,
        filename="patient_t1_axial.png",
        model_id="efficientnetv2b0",
        model_name="EfficientNetV2B0",
    )
    assert analysis_id.startswith("ANL-")

    res = await client.get(f"/api/analyses/{analysis_id}", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["analysis_id"] == analysis_id
    assert data["predictedClass"] == "Glioma"
    assert data["confidence"] == 0.942
    assert len(data["scores"]) == 4


@pytest.mark.asyncio
async def test_performance_endpoints(client: AsyncClient):
    # 1. Classification performance
    cls_res = await client.get("/api/performance/classification")
    assert cls_res.status_code == 200
    cls_data = cls_res.json()
    models = {m["model_id"]: m for m in cls_data["models"]}
    assert "efficientnetv2b0" in models
    assert "densenet121" in models
    assert "resnet50" in models

    assert models["efficientnetv2b0"]["evaluated"] is True
    assert models["efficientnetv2b0"]["accuracy"] == 0.941
    assert models["densenet121"]["evaluated"] is False
    assert models["densenet121"]["status"] == "Not evaluated"
    assert models["resnet50"]["evaluated"] is False
    assert models["resnet50"]["status"] == "Not evaluated"

    # 2. Segmentation performance
    seg_res = await client.get("/api/performance/segmentation")
    assert seg_res.status_code == 200
    seg_data = seg_res.json()
    seg_models = {m["model_id"]: m for m in seg_data["models"]}
    assert "attention_cnn_vit" in seg_models
    assert "brats_3d_unet" in seg_models
    assert "attention_unet" in seg_models
    assert "unet_plus_plus" in seg_models

    assert seg_models["attention_cnn_vit"]["evaluated"] is True
    assert seg_models["attention_cnn_vit"]["dice_coefficient"] == 0.8937
    assert seg_models["attention_cnn_vit"]["mean_iou"] == 0.8078

    assert seg_models["brats_3d_unet"]["evaluated"] is True
    assert seg_models["brats_3d_unet"]["dice_coefficient"] == 0.82976
    assert seg_models["brats_3d_unet"]["pixel_accuracy"] == 0.995746

    assert seg_models["attention_unet"]["evaluated"] is False
    assert seg_models["attention_unet"]["status"] == "Not evaluated"
    assert seg_models["unet_plus_plus"]["evaluated"] is False
    assert seg_models["unet_plus_plus"]["status"] == "Not evaluated"

    # 3. Model comparison
    comp_res = await client.get("/api/performance/compare")
    assert comp_res.status_code == 200
    comp_data = comp_res.json()
    assert "classification_comparison" in comp_data
    assert "segmentation_comparison" in comp_data

    # 4. Unified models list
    models_res = await client.get("/api/performance/models")
    assert models_res.status_code == 200
    assert len(models_res.json()) >= 7

    # 5. Historical evaluation runs
    hist_res = await client.get("/api/performance/evaluation-history")
    assert hist_res.status_code == 200
    assert len(hist_res.json()) > 0
