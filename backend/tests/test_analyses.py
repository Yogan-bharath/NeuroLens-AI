import json
import pytest
from httpx import AsyncClient
from app.utils.serialization import image_to_base64_data_url
from app.core.exceptions import ModelUnavailableException
from app.services.explainability_service import explainability_service
from app.services.segmentation_service import segmentation_service
from PIL import Image
import numpy as np


@pytest.mark.asyncio
async def test_unverified_attention_vit_inference_is_rejected():
    with pytest.raises(ModelUnavailableException):
        await segmentation_service.segment(
            image_bytes=b"",
            analysis_id="test",
            model_id="attention_vit_segmentation",
        )

    with pytest.raises(ModelUnavailableException):
        await explainability_service.generate_explanation(
            image_bytes=b"",
            analysis_id="test",
            method="seg_gradcam",
            model_id="attention_vit_segmentation",
        )


@pytest.mark.asyncio
async def test_analyses_endpoints(client: AsyncClient, synthetic_mri_bytes: bytes, auth_headers: dict):
    # 1. Multipart Combined Analysis
    files = {"file": ("patient_scan.png", synthetic_mri_bytes, "image/png")}
    config_dict = {
        "mode": "combined",
        "classificationModelId": "efficientnetv2b0",
        "segmentationModelId": "unet",
        "classificationXaiMethod": "gradcam",
        "classificationLayer": "top_conv"
    }
    form_data = {"config": json.dumps(config_dict)}

    res = await client.post("/api/analyses", files=files, data=form_data, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert "analysis_id" in data
    assert data["status"] == "completed"
    assert data["analysis_mode"] == "combined"
    assert data["classification"] is not None
    assert len(data["classification"]) == 4
    assert data["segmentation"] is not None
    assert len(data["segmentation"]["classes"]) >= 1
    assert data["segmentation"]["classes"][0]["label"] == "Tumor Lesion"
    assert len(data["explanations"]) >= 1
    assert "disclaimer" in data

    analysis_id = data["analysis_id"]

    # 2. Get Analysis by ID
    get_res = await client.get(f"/api/analyses/{analysis_id}", headers=auth_headers)
    assert get_res.status_code == 200
    assert get_res.json()["analysis_id"] == analysis_id

    # 3. Check status endpoint
    status_res = await client.get(f"/api/analyses/{analysis_id}/status", headers=auth_headers)
    assert status_res.status_code == 200
    assert status_res.json()["status"] == "completed"

    # 4. JSON Base64 Analysis (used by frontend api.ts)
    raw_img = Image.fromarray(np.zeros((256, 256), dtype=np.uint8), mode="L")
    b64_url = image_to_base64_data_url(raw_img)
    json_payload = {
        "image": b64_url,
        "fileName": "frontend_scan.png",
        "config": {
            "mode": "classification",
            "classificationModelId": "efficientnetv2b0"
        }
    }
    json_res = await client.post("/api/analyses", json=json_payload, headers=auth_headers)
    assert json_res.status_code == 200
    assert json_res.json()["analysis_mode"] == "classification"

    # 5. Frontend V1 Compatibility route /api/v1/analyze
    v1_res = await client.post("/api/v1/analyze", json={
        "image": b64_url,
        "fileName": "v1_scan.png"
    }, headers=auth_headers)
    assert v1_res.status_code == 200
    v1_data = v1_res.json()
    assert "id" in v1_data

    # 6. Retry Analysis
    retry_res = await client.post(f"/api/analyses/{analysis_id}/retry", headers=auth_headers)
    assert retry_res.status_code == 200

    # 7. Delete Analysis
    del_res = await client.delete(f"/api/analyses/{analysis_id}", headers=auth_headers)
    assert del_res.status_code == 200
