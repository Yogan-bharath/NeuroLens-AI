import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_chat_service(client: AsyncClient):
    payload = {"message": "Why did the model predict this class?"}
    res = await client.post("/api/v1/chat", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["role"] == "assistant"
    assert "Important:" in data["content"] or "neuroradiologist" in data["content"]
