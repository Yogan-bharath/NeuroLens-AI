# NeuroScan Integration Reconnaissance Report (Phase 0)

## 1. System Architecture & Execution Model
- **Backend Stack**: FastAPI with Uvicorn ASGI server (`uvicorn app.main:app`), running as a long-running stateful process. Not serverless.
- **Python / Runtime**: Python 3.11.9 inside `backend/.venv`.
- **Framework Versions**:
  - `TensorFlow`: 2.21.0
  - `Keras`: 3.15.1
  - `Pytest`: 9.1.1
  - `NumPy`: 1.26.4
  - `Nibabel`: 5.2.1

## 2. Model Registration & Loading
- **Current Loader**: `backend/app/ml/model_loader.py` maintains `ModelLoader` with in-memory caches `_classification_models` and `_segmentation_models`.
- **Current Catalog**: `backend/app/ml/model_registry.py` defines metadata records returned by `GET /api/models` and `GET /api/classification/models`:
  - `efficientnetv2b0`: Currently points to `EfficientNetV2B0_clahe_iid_best.keras` (128×128×3 RGB input).
  - `swin_transformer`: Points to `swin_transformer.keras` (224×224×3 RGB input, custom Swin Transformer layers).
  - `unet`: Points to `BRISC_Binary_UNet_best.keras` (128×128×3 RGB input).
  - `deeplabv3plus`: Points to `deeplabv3plus.keras` (256×256×3 RGB input).
  - `brats_3d_unet`: Loaded via `_load_brats_3d_unet` (points to `backend/model_artifacts/brats_3d_unet/best_3d_unet_model.keras`).
- **Target Design**: Modular lazy-loading registry in `backend/app/ml/unified_registry.py`. Models load on first demand and are cached in memory rather than all 5 loading synchronously at application startup.

## 3. Classification & Grad-CAM Pipelines
- **Endpoint**: `POST /api/classification/analyze` in `backend/app/api/routes/classification.py`.
- **Flow**:
  1. Image validation (PNG/JPG/JPEG, <= 15MB).
  2. Preprocessing via `preprocessing_service`:
     - EfficientNet (legacy): Bilinear resize to 128×128, float32 scaled to `[0, 1]` via `/ 255.0`, optional CLAHE.
     - Swin Transformer: Bilinear resize to 224×224, float32 scaled to `[0, 1]`.
  3. Inference: `model(batch, training=False)` producing 4-class softmax scores.
  4. Explainability via `explainability_service.generate_explanation()`:
     - Target layers: `top_conv` for EfficientNet; `swin_transformer_block_7` for Swin.
     - Implementation: `make_gradcam()` and `make_gradcam_plus_plus()` in `app/ml/inference_utils.py`.
     - Heatmap blended with original image (alpha 0.40) and stored via `imagekit_service` / local persistent disk storage.

## 4. 2D Segmentation & DeepLabV3+ XAI
- **Model Details (`deeplabv3plus.keras`)**:
  - Verified input shape: `(None, 256, 256, 3)` float32 in range `[0, 1]`.
  - Verified output shape: `(None, 256, 256, 1)` sigmoid probability map.
  - Total layers: 179 layers.
  - Verified candidate layers for Seg-Grad-CAM:
    - `conv2d_20`: ASPP projection layer, shape `(None, 16, 16, 256)`.
    - `up_sampling2d_2`: Decoder layer before final 4x upsampling, shape `(None, 64, 64, 256)`.
- **Seg-Grad-CAM Objective**:
  - Target score $S = \sum_{h,w} \hat{y}_{h,w}$ (sum of predicted foreground probabilities).
  - Channel weights: Global average of gradients $\frac{\partial S}{\partial A^k}$.
  - Heatmap: $\text{ReLU}\left(\sum_k w_k A^k\right)$, normalized to $[0, 1]$ by max value, resized to 256×256.

## 5. 3D Volumetric Segmentation & Workstation Contracts
- **Endpoints**:
  - `POST /api/segmentation/validate-study`: Validates 4 required sequences (FLAIR, T1, T1CE, T2), affine matrix alignment, voxel spacing, and dimensions.
  - `POST /api/segmentation/analyze` & `POST /api/segmentation/predict`: Volumetric sliding window inference.
  - `GET /api/segmentation/slice/{prediction_id}`: Synchronized 2D multiplanar rendering (Axial, Coronal, Sagittal) supporting overlay, mask-only, and original view modes.
  - `GET /api/segmentation/download/{prediction_id}`: NIfTI export (`.nii.gz`) preserving input FLAIR affine matrix and header.
- **Label Mapping & Colors**:
  - Label 0: Background (`#000000`, RGBA `(0, 0, 0, 0)`)
  - Label 1: Necrotic / Non-enhancing (`#EF4444`, RGBA `(239, 68, 68, 200)`)
  - Label 2: Peritumoral Edema (`#10B981`, RGBA `(16, 185, 129, 200)`)
  - Label 3: Enhancing Tumor (`#F59E0B`, RGBA `(245, 158, 11, 200)`) (BraTS 4 is remapped to 3 internally; exported NIfTI maps 3 back to 4).
- **3D Viewer Consumption**:
  - Multiplanar slices: PNG data URLs from `render_slice_at()`.
  - 3D Surface reconstruction: `generate_3d_surface_points()` downsamples tumor coordinates to centered $[-1, 1]$ points, plus brain boundary points from non-zero MRI tissue erosion.
  - Centroid & Stereotactic localization: `nib.affines.apply_affine` maps tumor voxels to patient RAS coordinates.
- **3D Grad-CAM Bug in Reference Notebook**:
  - `make_3d_gradcam` in the original notebook used `tf.image.resize` on a 3D volume, treating the first axis as batch and yielding an erroneous shape `(96, 96, 96)` instead of `(96, 96, 64)`.
  - Fix: Trilinear interpolation (`scipy.ndimage.zoom` with `order=1`) directly from bottleneck `(12, 12, 8)` to patch `(96, 96, 64)`.

## 6. UI Hardcoded Model Text Audit
The frontend contained hardcoded references in:
- `frontend/src/pages/ExplainabilityPage.tsx`:
  - `Feature Map: {isSwin ? '7 × 7 × 768' : '8 × 8 × 1280'}`
  - `LAYER: {isSwin ? 'SWIN_BLOCK_7' : 'TOP_CONV'}`
  - `'Attention CNN + ViT (BraTS)'`
- `frontend/src/pages/Classification2DResultsPage.tsx`:
  - Architecture and parameter string fallbacks (`28.3M params`, `7.11M params`).
- `frontend/src/pages/Segmentation3DResultsPage.tsx`:
  - Hardcoded banner: `"GENUINE MODEL PREDICTION · NO GRAD-CAM"`.
  - Hardcoded parameters: `"3.07M params"` vs `"16.4M params"`.

## 7. Model Artifact Audit
- `backend/model_artifacts/EfficientNetV2B0_clahe_256.keras`: Verified present (75.4 MB). Input: `(None, 256, 256, 3)` float32 0–255. Target layer: `top_activation` (`8 × 8 × 1280`).
- `backend/model_artifacts/deeplabv3plus.keras`: Verified present (71.9 MB). Input: `(None, 256, 256, 3)`. Layers `conv2d_20` and `up_sampling2d_2` verified.
- `backend/model_artifacts/brats_3d_unet/best_3d_unet_model.keras`: Present (12.6 MB). Identical in size to `neurolens_attention_vit.keras` (hybrid Attention CNN + ViT slice model). For genuine 96×96×64 3D U-Net (5.61M parameters), the stand-in architecture builder from `reference/tests/test_reference.py` is integrated, with lazy weight-loading and fallback as specified by Rule 4.

