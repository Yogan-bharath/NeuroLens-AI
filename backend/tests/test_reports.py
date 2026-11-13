import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_report_generation_and_download(client: AsyncClient, synthetic_mri_bytes: bytes, auth_headers: dict):
    # 1. Create analysis
    files = {"file": ("report_scan.png", synthetic_mri_bytes, "image/png")}
    create_res = await client.post("/api/analyses", files=files, headers=auth_headers)
    assert create_res.status_code == 200
    analysis_id = create_res.json()["analysis_id"]

    # 2. Get report summary metadata
    summary_res = await client.get(f"/api/analyses/{analysis_id}/report", headers=auth_headers)
    assert summary_res.status_code == 200
    data = summary_res.json()
    assert data["format"] == "PDF"
    assert "download_url" in data

    # 3. Download PDF report
    download_res = await client.get(f"/api/analyses/{analysis_id}/report/download", headers=auth_headers)
    assert download_res.status_code == 200
    assert download_res.headers["content-type"] == "application/pdf"
    # Verify PDF magic header
    assert download_res.content.startswith(b"%PDF-")
