# Seg-Grad-CAM 2D Tumor Segmentation Diagnosis & Fix Report

## 1. Executive Summary & Root Cause Analysis

### Observed Issue
In the Tumor Segmentation panel on the 2D classification page, selecting **Explain** mode displayed:
- The text label: `SEG-GRAD-CAM EXPLAIN, LAYER: conv2d_20 · MAP: 16 × 16 × 256`
- The caption: `Shows which features drove the total predicted tumor area`
- But the rendered image was the solid red filled mask with amber outline and `TUMOR AREA: 2.53%` (identical to Overlay mode), rather than a smooth blue-to-red Seg-Grad-CAM attribution heatmap.

### Root Cause Diagnosis (Options B + A)
1. **Root Cause B (Backend Missing / Incomplete Contract)**:
   - In `/api/classification/analyze`, `explain_segmentation` was a form parameter defaulting to `False`. The frontend upload page never provided this flag.
   - In `/api/classification/{analysis_id}/segment`, `segmentation_2d_service.segment_2d` was invoked without `explain=True`, so the endpoint response completely omitted explanation fields (`explain_url` / `explain.heatmap_png` were `None`/`undefined`).
2. **Root Cause A (Frontend Render Fallback)**:
   - In `frontend/src/pages/Classification2DResultsPage.tsx`, the `<img>` tag rendered:
     ```tsx
     src={segViewMode === 'explain' ? (analysisData.explain_segmentation_url || segmentationOverlayUrl) : ...}
     ```
     Because `explain_segmentation_url` was missing from the response, the expression fell back to `segmentationOverlayUrl` (the predicted mask overlay).
   - The HUD label had hardcoded fallbacks `'conv2d_20'` and `'16 × 16 × 256'` when data was absent.
   - The UI lacked a dedicated heatmap overlay component supporting the page's blend modes (`Screen`, `Overlay`, `Direct`), opacity slider, thin boundary outline, and colormap legend.
3. **Hypothesis C Disproven**:
   - The backend heatmap on `conv2d_20` is **NOT degenerate**. It generates strong, non-zero gradient activations across all tested real MRI images (max ~0.97 - 0.995).

---

## 2. Empirical Verification Across Real Patient MRI Scans

Using `scripts/check_seg_explain.py`, the real `backend/model_artifacts/deeplabv3plus.keras` model was executed across 5 real patient scans from `backend/storage/uploads/` on layer `conv2d_20` (output shape `16 × 16 × 256`):

| Image Filename | Min | Max | Mean | Hot Fraction (>0.5) | Heat Mass In Mask | Degenerate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `0b938fbd0afa3b33_7823ad.jpg` | 0.000 | 0.994 | 0.055 | 1.10% | 19.51% | **False** |
| `0fe64f57b8cc5c8e_620829.jpg` | 0.000 | 0.987 | 0.025 | 1.02% | 48.32% | **False** |
| `2575cc82469af937_cd816b.jpg` | 0.000 | 0.979 | 0.037 | 0.47% | 14.08% | **False** |
| `3aae945e9baaea41_74dccb.jpg` | 0.000 | 0.996 | 0.133 | 2.88% | 12.40% | **False** |
| `428de1d6127d05e9_626997.jpg` | 0.000 | 0.970 | 0.010 | 0.33% | 16.83% | **False** |

### Fine Layer Check (`up_sampling2d_2`)
- **Feature Map Shape:** `(256, 256, 256)`
- **Stats:** `min=0.0`, `max=0.999999`, `mean=0.0122`, `hot_fraction=1.41%`, `degenerate=False`.
- **Finding:** Both `conv2d_20` (deep multiscale context) and `up_sampling2d_2` (fine boundary features) produce non-degenerate maps.

### Saved Debug Artifacts (under `backend/storage/debug/`)
- `sample_0_0b938fbd0afa3b33_7823ad_raw_cam.png`
- `sample_0_0b938fbd0afa3b33_7823ad_heatmap_rgba.png`
- `sample_0_0b938fbd0afa3b33_7823ad_composite.png`
- `sample_1_0fe64f57b8cc5c8e_620829_composite.png`
- `sample_2_2575cc82469af937_cd816b_composite.png`
- `sample_3_3aae945e9baaea41_74dccb_composite.png`
- `sample_4_428de1d6127d05e9_626997_composite.png`

---

## 3. Implementation Details

### Backend Additions & Changes
1. **New Module `backend/app/ml/seg2d_explain.py`**:
   - `seg_gradcam`: GradientTape attribution on sum of predicted tumor probabilities over all pixels.
   - `heatmap_rgba`: Jet colormap with linear alpha ramp (cold pixels transparent, hot pixels opaque red).
   - `encode_png_rgba`: Encodes transparent RGBA PNG.
   - `heat_stats`: Computes min, max, mean, hot fraction, heat mass in mask, and degenerate flag.
   - `generate_segmentation_explanation`: Orchestrates computation and returns structured payload.
2. **Internal ML Modules Added to `backend/app/ml/`**:
   - `classifier_efficientnet256.py`: Standalone math for `EfficientNetV2B0_clahe_256.keras`.
   - `seg2d_xai.py`: Layer resolver and Seg-Grad-CAM helper.
   - `unet3d.py`: 3D volumetric sliding-window and trilinear Grad-CAM logic.
   - *Fix applied:* Eliminates `No module named 'reference'` import crashes when running under uvicorn.
3. **Endpoint Hooks**:
   - `backend/app/services/segmentation_2d_service.py`: Computes Seg-Grad-CAM explanation and populates `explain` object and `explain_url`.
   - `backend/app/api/routes/classification.py`: Both `/analyze` and `/{analysis_id}/segment` default `explain=True`, save explanation to DB/cache, and return all explanation fields.

### Frontend Additions & Changes
1. **New Component `frontend/src/components/analysis/SegExplainViewer.tsx`**:
   - Renders smooth Jet heatmap PNG overlaid on the base MRI using the active blend mode (`Screen`, `Overlay`, `Direct`) and opacity slider.
   - Draws **NO filled mask**.
   - Includes thin **Tumor Outline** contour toggle (amber `#F59E0B`, default ON).
   - Displays Jet colormap legend bar: `0.0 (Suppressed)` $\longrightarrow$ `1.0 (Peak Saliency)`.
   - Includes layer selector: `Deep (conv2d_20, 16×16)` and `Fine (up_sampling2d_2, 256×256)`.
   - Shows fallback alert if `available === false`: *"No explanation available for this layer"*.
   - Displays required disclaimers:
     > *"Shows which features drove the total predicted tumor area. The score is the predicted tumor area, so hot regions usually overlap the mask; the map is coarse (16 x 16) and is not a tumor boundary."*
     > *"Research/educational use only. Not clinically validated."*
2. **Model Registry Cleanup**:
   - In `frontend/src/services/modelRegistry.ts`, `efficientnetv2b0` (legacy 128x128) is marked `status: 'unavailable'` and replaced by `EfficientNetV2B0 (CLAHE, 256x256)` as the active primary classifier.
3. **Page Hook**:
   - `Classification2DResultsPage.tsx`: Mounts `SegExplainViewer` when `segViewMode === 'explain'`. Keeps `Overlay` and `Binary Mask` rendering identical to previous behavior.

---

## 4. Verification & Test Results

1. **Python Unit Tests (`tests/test_seg_explain.py`)**:
   - `test_heatmap_rgba_properties`: **PASSED** (cold alpha=0, hot alpha=255 red-dominant, PNG round-trip verified).
   - `test_heat_stats_degenerate_and_mask_fraction`: **PASSED** (all-zero and constant degenerate detection, heat mass in mask = 1.0).
   - `test_seg_gradcam_on_standin_model`: **PASSED** (stand-in Conv2D model, shape `(48, 48)`, range in `[0, 1]`, no NaNs).
   - `test_generate_segmentation_explanation_contract`: **PASSED** (full payload schema).
   - `test_segmentation_service_explain_integration`: **PASSED** (mocked model integration, non-constant alpha channel verified).
2. **Live Model Execution**:
   - `backend/model_artifacts/EfficientNetV2B0_clahe_256.keras`: Verified live on real MRI scan. Output: `predicted_class: 'Meningioma'`, `confidence: 0.99995`, layer `top_activation` shape `(8, 8, 1280)`.
3. **Frontend Build Verification**:
   - `npm run build` (`tsc && vite build`): **PASSED** with 0 errors.
