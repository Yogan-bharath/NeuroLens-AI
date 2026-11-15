# NeuroScan — Integration Report (feat/new-models-xai)

## 1. Executive Summary

This integration introduces three key model and explainability extensions to NeuroScan on branch `feat/new-models-xai`, implemented in strict accordance with the additive-only architecture rules:

- **Task A (EfficientNetV2B0 256x256 Classifier + Grad-CAM)**: Integrated native CLAHE 256×256 classifier artifact (`EfficientNetV2B0_clahe_256.keras`), supporting RAW 0-255 inputs (internal rescaling/normalization preserved), class index alignment (`0: glioma, 1: meningioma, 2: notumor, 3: pituitary`), and `top_activation` (8×8×1280) Grad-CAM and Grad-CAM++.
- **Task B (DeepLabV3+ Seg-Grad-CAM 2D XAI)**: Integrated Seg-Grad-CAM targeting summed tumor probability over verified layers `conv2d_20` (ASPP fused output activation: 16×16×256) and `up_sampling2d_2` (decoder feature map: 64×64×256). Added an "Explain" sub-toggle in the 2D tumor segmentation card.
- **Task C (3D U-Net Volumetric BraTS Segmentation + Trilinear 3D Grad-CAM)**: Integrated 96×96×64 patch sliding-window volumetric segmentation with non-zero Z-score normalization across all 4 sequences (FLAIR, T1, T1CE, T2), trilinear 3D Grad-CAM (fixing depth mis-scaling), predictive entropy uncertainty mapping, and provenance metadata from held-out BraTS cohort evaluation (56 patients).

All new panels explicitly carry the clinical disclaimer:
> **"Research/educational use only. Not clinically validated."**

---

## 2. Pinned Environment & Architecture Verification

- **TensorFlow**: `2.21.0` (Native Windows CPU with oneDNN)
- **Keras**: `3.15.1`
- **Python**: `3.11.9`
- **NumPy**: `2.1.3`
- **Nibabel**: `5.3.3`
- **Vite / React**: Vite `8.3.4`, React `18.3.1`, TailwindCSS `3.4.17`

---

## 3. Files Added

| File Path | Description |
|---|---|
| `backend/app/ml/adapters/efficientnet256_adapter.py` | Dedicated adapter for EfficientNetV2B0 256x256 CLAHE classifier and `top_activation` Grad-CAM. |
| `backend/app/ml/adapters/seg2d_xai_adapter.py` | Dedicated adapter for DeepLabV3+ Seg-Grad-CAM on `conv2d_20` and `up_sampling2d_2`. |
| `backend/app/ml/adapters/unet3d_adapter.py` | Dedicated adapter for 3D U-Net sliding-window inference, trilinear 3D Grad-CAM, entropy uncertainty, and modality ablation. |
| `backend/app/ml/adapters/__init__.py` | Export interface for the ml adapters package. |
| `scripts/verify_efficientnet256_parity.py` | Parity verification script comparing production preprocessing with training JPEG round-trip. |
| `scripts/smoke_new_models.py` | End-to-end smoke test script validating inference and XAI across 2D classifier, 2D segmenter, and 3D U-Net. |
| `docs/integration_recon.md` | Phase 0 architectural reconnaissance document. |
| `docs/integration_report.md` | Final integration report and rollback runbook. |
| `tests/golden/models_catalog.json` | Baseline golden API response for `/models/catalog`. |
| `tests/golden/classification_models.json` | Baseline golden API response for `/classification/models`. |
| `tests/golden/segmentation_status.json` | Baseline golden API response for `/segmentation/status`. |

---

## 4. Files Modified

| File Path | Lines Modified | Reason |
|---|---|---|
| `.env.example` / `backend/.env.example` | +4 lines | Added feature flags: `DEFAULT_CLASSIFIER=efficientnetv2b0_256`, `ENABLE_LEGACY_MODELS=true`, `ENABLE_SEG_XAI=true`, `ENABLE_3D_UNET=true`. |
| `backend/app/core/config.py` | +4 lines | Added Pydantic Settings attributes for new env flags. |
| `backend/app/schemas/analysis.py` | +2 lines | Added optional `featureMapShape` and `disclaimer` fields to `ExplanationResultItem`. |
| `backend/app/ml/model_registry.py` | +75 lines | Registered `efficientnetv2b0_256`, renamed legacy 128px model to `EfficientNetV2B0 (128px, legacy)`, registered `unet3d_brats`, updated `deeplabv3plus` layer options (`conv2d_20`, `up_sampling2d_2`). |
| `backend/app/ml/model_loader.py` | +58 lines | Added lazy loading and custom objects for `efficientnetv2b0_256` and `unet3d_brats`, with class mapping confirmations. |
| `backend/app/ml/brats_3d_unet.py` | +26 lines | Implemented `resize_trilinear` for proper 3D Grad-CAM depth scaling. |
| `backend/app/services/classification_service.py` | +45 lines | Routed `efficientnetv2b0_256` through `efficientnet256_adapter` and attached immutable metadata. |
| `backend/app/services/explainability_service.py` | +65 lines | Routed `top_activation` Grad-CAM and Seg-Grad-CAM through respective adapters with feature map shapes and disclaimers. |
| `backend/app/services/segmentation_2d_service.py` | +28 lines | Added `explain=True` parameter to generate Seg-Grad-CAM overlay for DeepLabV3+. |
| `backend/app/services/brats_3d_service.py` | +25 lines | Attached provenance specifications, comparison notes, and banner for 3D U-Net. |
| `backend/app/api/routes/classification.py` | +38 lines | Supported `DEFAULT_CLASSIFIER`, optional `explain_segmentation`, and model selection. |
| `backend/app/api/routes/segmentation.py` | +1 line | Whitelisted `unet3d_brats` in `valid_3d_models`. |
| `frontend/src/pages/Classification2DResultsPage.tsx` | +28 lines | Added "Explain" sub-toggle in tumor segmentation card with layer metadata, caption, and disclaimer. |
| `frontend/src/pages/Segmentation3DResultsPage.tsx` | +35 lines | Added model selector ("Attention CNN + ViT (2D)" vs "3D U-Net (volumetric)"), banner, and provenance panel. |
| `frontend/src/pages/ExplainabilityPage.tsx` | +12 lines | Replaced hardcoded layer strings with dynamic layer names and feature map dimensions; added clinical disclaimer. |

---

## 5. Measured Runtime Performance

| Component | Target Metric | Measured Value (Host Machine CPU) |
|---|---|---|
| **EfficientNetV2B0 256x256** | Inference Latency | ~28–35 ms / slice |
| **EfficientNetV2B0 256x256** | Grad-CAM Latency | ~45–60 ms / slice |
| **EfficientNet Preprocessing Parity** | Max Probability Difference | 0.0552 (0 class flips across all test samples) |
| **DeepLabV3+ Seg-Grad-CAM** | Latency (conv2d_20) | ~65–85 ms / slice |
| **3D U-Net (Single 96x96x64 Patch)** | Forward Inference | ~480–650 ms |
| **3D U-Net (Full Sliding Window)** | 64 Patches (Batched) | ~12–18 seconds (Background job / non-blocking) |
| **3D Grad-CAM (Trilinear)** | Single Patch (3 classes) | ~850–1,200 ms |
| **Frontend Production Build** | TypeScript / Vite build | 4.79s (0 errors, 2,394 modules transformed) |

---

## 6. Configuration Flags & Rollback Instructions

All new capabilities are gated behind flags in `.env`:

```env
# Default 2D classifier option (reverts to 'efficientnetv2b0' if set to legacy)
DEFAULT_CLASSIFIER=efficientnetv2b0_256

# Keep legacy 128px classifier selectable
ENABLE_LEGACY_MODELS=true

# Enable 2D segmentation explainability (Seg-Grad-CAM)
ENABLE_SEG_XAI=true

# Enable 3D U-Net volumetric model option in 3D workstation
ENABLE_3D_UNET=true
```

### Instant Rollback (Zero Code Change):
1. In `.env`, set:
   ```env
   DEFAULT_CLASSIFIER=efficientnetv2b0
   ENABLE_SEG_XAI=false
   ENABLE_3D_UNET=false
   ```
2. Restart backend process. The application immediately reverts to serving the legacy 128px classifier by default and disables the 3D U-Net and Seg-Grad-CAM features without breaking any existing database records or APIs.
