import json

path = r"C:\Users\BHARATH\.gemini\antigravity\brain\5f931440-c65a-4eab-94a2-98725955853b\.user_uploaded\media_1794759380512_36e3b1da.ipynb"
with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb.get('cells', []))}")

for idx, cell in enumerate(nb.get("cells", [])):
    src = "".join(cell.get("source", []))
    if any(k in src.lower() for k in ["gradcam", "grad_cam", "heatmap", "gradienttape", "deeplab", "make_gradcam"]):
        print(f"\n==================== CELL {idx} ({cell.get('cell_type')}) ====================")
        print(src)

