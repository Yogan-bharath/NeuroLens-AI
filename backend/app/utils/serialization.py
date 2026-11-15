import base64
import io
from typing import Any, Dict, Optional
from bson import ObjectId
import numpy as np
from PIL import Image


def mongo_to_dict(doc: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Convert MongoDB document with ObjectIds to JSON-serializable dict."""
    if doc is None:
        return None
    res = {}
    for k, v in doc.items():
        if isinstance(v, ObjectId):
            res[k] = str(v)
            if k == "_id" and "id" not in res:
                res["id"] = str(v)
        elif isinstance(v, dict):
            res[k] = mongo_to_dict(v)
        elif isinstance(v, list):
            res[k] = [mongo_to_dict(item) if isinstance(item, dict) else item for item in v]
        else:
            res[k] = v
    if "_id" in res and "id" not in res:
        res["id"] = str(res["_id"])
    return res


def image_to_base64_data_url(img: Image.Image, format: str = "PNG") -> str:
    """Convert PIL image to base64 Data URL."""
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    mime = "image/png" if format.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def base64_data_url_to_bytes(data_url: str) -> bytes:
    """Extract bytes from a base64 Data URL, raw base64 string, storage URL, or remote HTTP(S) URL."""
    if not data_url:
        return b""
    if data_url.startswith("/api/storage/"):
        from app.services.storage_service import storage_service
        parts = data_url.strip("/").split("/")
        if len(parts) >= 4:
            category, filename = parts[2], parts[3]
            disk_path = storage_service.get_file_path(category, filename)
            if disk_path.exists():
                return disk_path.read_bytes()
    if data_url.startswith("http://") or data_url.startswith("https://"):
        import urllib.request
        req = urllib.request.Request(data_url, headers={"User-Agent": "NeuroScan-Backend/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.read()
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    return base64.b64decode(data_url)


def numpy_to_native(val: Any) -> Any:
    """Convert numpy scalar types to native python types."""
    if isinstance(val, (np.floating, float)):
        return float(val)
    elif isinstance(val, (np.integer, int)):
        return int(val)
    elif isinstance(val, np.ndarray):
        return val.tolist()
    elif isinstance(val, dict):
        return {k: numpy_to_native(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [numpy_to_native(v) for v in val]
    return val
