# NeuroScan Reference Components

Code extracted from the notebooks, with tests. Copy this folder into the repo as `reference/` (read-only).

| File | What it gives the integration |
|---|---|
| `classifier_efficientnet256.py` | Task A: `preprocess` (CLAHE -> 256x256, raw 0-255), `predict`, `gradcam` (layer `top_activation`), `feature_map_shape`, `CLASS_NAMES` |
| `seg2d_xai.py` | Task B: `resolve_layers` (notebook names first, else by role/shape), `seg_gradcam` (target = sum of predicted tumor probability) |
| `unet3d.py` | Task C: `prepare_volume`, `sliding_window_predict`, `gradcam3d` (resize FIXED), `resize_trilinear`, `prediction_entropy`, `tumor_center_start` / `extract_patch` / `embed_patch_map`, `CUSTOM_OBJECTS` |
| `metadata.json` | Numbers and protocols for the provenance panels |
| `tests/test_reference.py` | 16 tests using stand-in models (no weights needed) |

Run: `python -m pytest reference/tests -q`  (needs tensorflow + numpy + opencv-python; nibabel only if you call `load_nifti`)

## What the tests verify (TF 2.21 / Keras 3.15, random weights)
- CLAHE preprocessing output shape/range, and agreement with the training pipeline to within JPEG decoder noise (mean ~0.6 of 255).
- Grad-CAM outputs: correct shape, range [0,1], deterministic, no NaN; 3D map is (96,96,64).
- Sliding window equals full-volume prediction exactly for a pointwise model (including padding for small volumes).
- The notebook's 3D Grad-CAM resize returns (96,96,96) for a 64-deep patch; `resize_trilinear` returns (96,96,64) and keeps depth alignment.
- DeepLabV3+ layer resolution by role works on the real architecture (picks ASPP output 16x16x256 and decoder 64x64x256).

## What they do NOT verify
- Accuracy, or whether heatmaps look sensible: that needs the real weights and real images.
- That the saved `deeplabv3plus.keras` has the same layer names as a fresh build: `resolve_layers` handles both cases but print its result and check it.
- Parity between the app's own preprocessing and `classifier_efficientnet256.preprocess` on real uploads.
