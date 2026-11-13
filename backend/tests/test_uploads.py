import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_mri_upload_valid_and_invalid(client: AsyncClient, synthetic_mri_bytes: bytes, auth_headers: dict):
    # 1. Valid upload
    files = {"file": ("test_brain_mri.png", synthetic_mri_bytes, "image/png")}
    res = await client.post("/api/uploads/mri", files=files, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert "upload_id" in data
    assert data["width"] == 256
    assert data["height"] == 256
    assert data["mime_type"] == "image/png"

    # 2. Reject non-image file
    bad_files = {"file": ("malicious.txt", b"This is not a brain MRI", "text/plain")}
    bad_res = await client.post("/api/uploads/mri", files=bad_files, headers=auth_headers)
    assert bad_res.status_code == 422 or bad_res.status_code == 400
