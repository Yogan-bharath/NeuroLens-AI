import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_explainability_endpoints(client: AsyncClient, synthetic_mri_bytes: bytes, auth_headers: dict):
    # 1. Create analysis first
    files = {"file": ("xai_scan.png", synthetic_mri_bytes, "image/png")}
    create_res = await client.post("/api/analyses", files=files, headers=auth_headers)
    assert create_res.status_code == 200
    analysis_id = create_res.json()["analysis_id"]

    # 2. Generate Grad-CAM explanation on-demand
    exp_payload = {
        "method": "gradcam",
        "target_layer": "top_conv",
        "target_class": "Glioma"
    }
    exp_res = await client.post(f"/api/analyses/{analysis_id}/explain", json=exp_payload, headers=auth_headers)
    assert exp_res.status_code == 200
    exp_data = exp_res.json()
    assert exp_data["method"] == "gradcam"
    assert exp_data["heatmap_url"] is not None
    assert exp_data["overlay_url"] is not None

    # 3. Retrieve all explanations
    all_res = await client.get(f"/api/analyses/{analysis_id}/explanations", headers=auth_headers)
    assert all_res.status_code == 200
    assert len(all_res.json()) >= 1
