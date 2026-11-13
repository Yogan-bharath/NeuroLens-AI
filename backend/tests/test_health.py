import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_liveness(client: AsyncClient):
    response = await client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "service" in data


@pytest.mark.asyncio
async def test_health_readiness(client: AsyncClient):
    response = await client.get("/api/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "models_ready" in data
    assert "storage_ready" in data
