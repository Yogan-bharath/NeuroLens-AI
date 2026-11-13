import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_user_registration_and_login_flow(client: AsyncClient):
    # 1. Register
    reg_payload = {
        "name": "Neuro Researcher",
        "email": "neuro.test@example.com",
        "password": "SecurePassword123!",
        "role": "researcher"
    }
    reg_res = await client.post("/api/auth/register", json=reg_payload)
    assert reg_res.status_code == 200
    reg_data = reg_res.json()
    assert "access_token" in reg_data
    assert reg_data["user"]["email"] == "neuro.test@example.com"
    # Ensure password hash is NEVER exposed
    assert "password_hash" not in reg_data["user"]
    assert "password" not in reg_data["user"]

    # 2. Reject duplicate email
    dup_res = await client.post("/api/auth/register", json=reg_payload)
    assert dup_res.status_code == 400

    # 3. Login with correct password
    login_payload = {
        "email": "neuro.test@example.com",
        "password": "SecurePassword123!"
    }
    login_res = await client.post("/api/auth/login", json=login_payload)
    assert login_res.status_code == 200
    login_data = login_res.json()
    token = login_data["access_token"]

    # 4. Login with incorrect password
    bad_login = await client.post("/api/auth/login", json={
        "email": "neuro.test@example.com",
        "password": "WrongPassword!"
    })
    assert bad_login.status_code == 401

    # 5. Access /api/auth/me with Bearer token
    me_res = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me_res.status_code == 200
    me_data = me_res.json()
    assert me_data["email"] == "neuro.test@example.com"

    # 6. Reject unauthenticated access to /api/auth/me
    client.cookies.clear()
    unauth_res = await client.get("/api/auth/me")
    assert unauth_res.status_code == 401
