import json
import os

downloads_dir = r"C:\Users\BHARATH\Downloads"

def dump_cells_matching(filename, terms):
    path = os.path.join(downloads_dir, filename)
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"\n==========================================")
    print(f"NOTEBOOK: {filename}")
    print(f"==========================================")
    for i, cell in enumerate(nb["cells"][:15]):
        src = "".join(cell.get("source", []))
        if any(t.lower() in src.lower() for t in terms):
            print(f"--- Cell {i} ---")
            print(src[:600])

dump_cells_matching("DenseNet121_Brain_Tumor_Classification.ipynb", ["dataset", "class", "label", "split", "iid", "clahe"])
dump_cells_matching("DenseNet121_Brain_Tumor_Classification_Kaggle.ipynb", ["dataset", "class", "label", "split", "iid", "clahe"])
