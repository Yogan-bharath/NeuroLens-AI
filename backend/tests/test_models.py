import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_models_registry_endpoints(client: AsyncClient):
    # 1. Models catalog
    res = await client.get("/api/models")
    assert res.status_code == 200
    data = res.json()
    assert "classification" in data
    assert "segmentation" in data
    assert "xai_methods" in data

    # 2. Classification models
    cls_res = await client.get("/api/models/classification")
    assert cls_res.status_code == 200
    cls_models = cls_res.json()
    eff_model = next((m for m in cls_models if m["id"] == "efficientnetv2b0"), None)
    assert eff_model is not None
    assert eff_model["status"] == "available"
    assert "Glioma" in eff_model["outputClasses"]

    # 3. Segmentation models
    seg_res = await client.get("/api/models/segmentation")
    assert seg_res.status_code == 200
    seg_models = seg_res.json()
    unet = next((m for m in seg_models if m["id"] == "unet"), None)
    assert unet is not None
    assert "Tumor Lesion" in unet["classes"]

    brats_3d_unet = next((m for m in seg_models if m["id"] == "brats_3d_unet"), None)
    assert brats_3d_unet is not None
    assert brats_3d_unet["status"] in ("available", "coming_soon")
    assert brats_3d_unet["classes"] == [
        "Background",
        "Necrotic / non-enhancing tumor",
        "Peritumoral edema",
        "Enhancing tumor",
    ]

    attention_vit = next((m for m in seg_models if m["id"] == "attention_vit_segmentation"), None)
    assert attention_vit is not None
    assert attention_vit["status"] in ("available", "coming_soon")
    assert len(attention_vit["classes"]) == 4

    # 4. XAI methods
    xai_res = await client.get("/api/models/xai-methods")
    assert xai_res.status_code == 200
    xai_methods = xai_res.json()
    method_ids = [m["id"] for m in xai_methods]
    assert "gradcam" in method_ids
    assert "gradcam_pp" in method_ids
    assert "seg_gradcam" in method_ids

    # 5. Model details
    detail_res = await client.get("/api/models/efficientnetv2b0")
    assert detail_res.status_code == 200
    assert detail_res.json()["name"] == "EfficientNetV2B0"

    # 6. Invalid model 404
    bad_res = await client.get("/api/models/non_existent_model")
    assert bad_res.status_code == 404

    # 7. Model comparison
    comp_res = await client.get("/api/models/compare")
    assert comp_res.status_code == 200
    assert "classification" in comp_res.json()


@pytest.mark.asyncio
async def test_brats_3d_inference_stays_unavailable_without_checkpoint(
    client: AsyncClient,
    auth_headers: dict,
):
    from app.ml.model_loader import model_loader
    files = {
        "flair": ("flair.nii", b"not read while model is unavailable", "application/octet-stream"),
        "t1": ("t1.nii", b"not read while model is unavailable", "application/octet-stream"),
        "t1ce": ("t1ce.nii", b"not read while model is unavailable", "application/octet-stream"),
        "t2": ("t2.nii", b"not read while model is unavailable", "application/octet-stream"),
    }

    response = await client.post(
        "/api/segmentation/brats-3d-unet",
        files=files,
        headers=auth_headers,
    )

    if not model_loader.is_available("brats_3d_unet"):
        assert response.status_code == 503
        assert response.json()["error_code"] == "MODEL_UNAVAILABLE"
    else:
        assert response.status_code in (400, 422, 500)
