import pytest
from httpx import AsyncClient
from app.core.security import create_access_token


@pytest.mark.asyncio
async def test_history_and_user_isolation(client: AsyncClient, synthetic_mri_bytes: bytes, auth_headers: dict):
    # 1. User A creates analysis
    files = {"file": ("user_a_scan.png", synthetic_mri_bytes, "image/png")}
    create_res = await client.post("/api/analyses", files=files, headers=auth_headers)
    assert create_res.status_code == 200
    analysis_id = create_res.json()["analysis_id"]

    # 2. User A retrieves history
    hist_res = await client.get("/api/history", headers=auth_headers)
    assert hist_res.status_code == 200
    items = hist_res.json()
    assert any(item["id"] == analysis_id for item in items)

    # 3. User B token
    user_b_token = create_access_token(
        subject="user_b_456",
        extra_claims={"email": "user_b@example.com", "name": "User B", "role": "researcher"}
    )
    user_b_headers = {"Authorization": f"Bearer {user_b_token}"}

    # 4. User B cannot access User A's analysis
    forbidden_res = await client.get(f"/api/analyses/{analysis_id}", headers=user_b_headers)
    assert forbidden_res.status_code == 403

    # 5. User A checks stats
    stats_res = await client.get("/api/history/stats", headers=auth_headers)
    assert stats_res.status_code == 200
    assert "total_analyses" in stats_res.json()
