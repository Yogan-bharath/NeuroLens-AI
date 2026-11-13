import io
import pytest
import pytest_asyncio
import numpy as np
from PIL import Image
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.core.security import create_access_token
from app.ml.model_loader import model_loader


def create_synthetic_test_mri_png(width: int = 256, height: int = 256) -> bytes:
    """Create valid PNG bytes representing a synthetic MRI scan."""
    img_array = np.zeros((height, width), dtype=np.uint8)
    y, x = np.ogrid[:height, :width]
    # Simulated brain skull/parenchyma
    center_y, center_x = height // 2, width // 2
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    img_array[dist < 100] = 80
    img_array[dist < 70] = 160
    # Simulated hyperintense lesion
    img_array[(x - 90) ** 2 + (y - 95) ** 2 < 20 ** 2] = 240

    img = Image.fromarray(img_array, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def synthetic_mri_bytes() -> bytes:
    return create_synthetic_test_mri_png(256, 256)


@pytest.fixture
def test_user_id() -> str:
    return "test_researcher_001"


@pytest.fixture
def auth_headers(test_user_id: str) -> dict:
    token = create_access_token(
        subject=test_user_id,
        extra_claims={
            "email": "tester@neurolens.ai",
            "name": "Dr. Test Researcher",
            "role": "researcher"
        }
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_headers() -> dict:
    token = create_access_token(
        subject="admin_user_001",
        extra_claims={
            "email": "admin@neurolens.ai",
            "name": "System Administrator",
            "role": "admin"
        }
    )
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture
async def client():
    model_loader.load_models()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac
