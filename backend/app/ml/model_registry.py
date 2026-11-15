from typing import List, Optional, Any
from app.models.model_registry import (
    ClassificationModelInfo,
    SegmentationModelInfo,
    XaiMethodInfo,
    FeatureLayerOption,
    ClassificationMetrics,
    SegmentationMetrics,
)
from app.ml.model_loader import model_loader

# ─── Static Verified Metadata (Aligned with Frontend Registry) ─────────────
CLASSIFICATION_MODELS: List[ClassificationModelInfo] = [
    ClassificationModelInfo(
        id="efficientnetv2b0_256",
        name="EfficientNetV2B0 (CLAHE, 256x256)",
        architecture="EfficientNetV2-B0 (Fused-MBConv + Squeeze-and-Excitation)",
        description="EfficientNetV2B0 trained with native CLAHE preprocessing at 256×256. Rescaling and ImageNet normalization are internal to the model.",
        task="classification",
        version="v2.5.0-clahe256",
        status="available",
        metrics=ClassificationMetrics(
            testAccuracy=0.9625,
            macroPrecision=0.962,
            macroRecall=0.962,
            macroF1=0.962,
            inferenceTimeMs=32,
            evaluationDataset="1,440-image held-out split (random per-image split, no patient IDs, no external validation)",
        ),
        supportedXaiMethods=["gradcam", "gradcam_pp"],
        compatibleLayers=[
            FeatureLayerOption(
                id="top_activation",
                name="top_activation",
                description="Final activation layer before global pooling (8 × 8 × 1280 feature map)",
                isVerified=True,
                depthDescription="Resolution: 8 × 8 × 1280",
            ),
        ],
        outputClasses=["Glioma", "Meningioma", "No Tumor", "Pituitary"],
        parametersCount="7.11M",
    ),
    ClassificationModelInfo(
        id="efficientnetv2b0",
        name="EfficientNetV2B0 (128px, legacy)",
        architecture="EfficientNetV2-B0 (Fused-MBConv)",
        description="A convolutional neural network optimized for four-class brain MRI classification with fast convergence and low parameter footprint.",
        task="classification",
        version="v2.4.0",
        status="available",
        metrics=ClassificationMetrics(
            testAccuracy=0.941,
            macroPrecision=0.938,
            macroRecall=0.935,
            macroF1=0.936,
            inferenceTimeMs=38,
            evaluationDataset="BraTS / Figshare Benchmark Test Split",
        ),
        supportedXaiMethods=["gradcam", "gradcam_pp", "integrated_gradients"],
        compatibleLayers=[
            FeatureLayerOption(
                id="top_conv",
                name="top_conv",
                description="Final 1×1 convolutional projection layer prior to global pooling",
                isVerified=True,
                depthDescription="Depth: Layer 168 · Resolution: 8 × 8",
            ),
            FeatureLayerOption(
                id="block4b_expand_activation",
                name="block4b_expand_activation",
                description="Intermediate stage expand activation capturing regional mid-level patterns",
                isVerified=True,
                depthDescription="Depth: Layer 112 · Resolution: 16 × 16",
            ),
            FeatureLayerOption(
                id="block3b_expand_activation",
                name="block3b_expand_activation",
                description="Early-mid layer capturing edge and texture representations",
                isVerified=False,
                depthDescription="Depth: Layer 74 · Resolution: 32 × 32",
            ),
        ],
        outputClasses=["Glioma", "Meningioma", "Pituitary", "No Tumor"],
        parametersCount="6.28M",
    ),
    ClassificationModelInfo(
        id="swin_transformer",
        name="Swin Transformer",
        architecture="Hierarchical Shifted-Window Transformer",
        description="Trained 224 × 224 RGB Swin Transformer classifier with transformer Grad-CAM.",
        task="classification",
        version="SwinTransformer-224",
        status="available",
        metrics=None,
        supportedXaiMethods=["gradcam"],
        compatibleLayers=[
            FeatureLayerOption(
                id="swin_transformer_block_7",
                name="swin_transformer_block_7",
                description="Final 7 × 7 shifted-window feature map before global pooling",
                isVerified=True,
                depthDescription="Resolution: 7 × 7",
            ),
        ],
        outputClasses=["Glioma", "Meningioma", "Pituitary", "No Tumor"],
        parametersCount="28.3M",
    ),
    ClassificationModelInfo(
        id="densenet121",
        name="DenseNet121",
        architecture="Densely Connected Convolutional Network 121",
        description="A densely connected convolutional architecture that connects each layer to every other layer in a feed-forward fashion.",
        task="classification",
        version="v1.1.0-rc",
        status="coming_soon",
        statusReason="Model checkpoint undergoing independent test split benchmark validation. Integration pending.",
        metrics=None,
        supportedXaiMethods=["gradcam", "integrated_gradients"],
        compatibleLayers=[
            FeatureLayerOption(
                id="features.denseblock4",
                name="features.denseblock4",
                description="Final dense block with dense concatenated feature channels",
                isVerified=True,
                depthDescription="Depth: Final Dense Block · Resolution: 8 × 8",
            ),
        ],
        outputClasses=["Glioma", "Meningioma", "Pituitary", "No Tumor"],
        parametersCount="8.0M",
    ),
    ClassificationModelInfo(
        id="resnet50",
        name="ResNet50",
        architecture="Residual Neural Network 50-Layer",
        description="A deep residual neural network architecture utilizing identity shortcut connections to mitigate vanishing gradient degradation.",
        task="classification",
        version="v2.1.0",
        status="coming_soon",
        statusReason="Residual Network weights undergoing evaluation benchmark.",
        metrics=None,
        supportedXaiMethods=["gradcam", "gradcam_pp", "integrated_gradients"],
        compatibleLayers=[
            FeatureLayerOption(
                id="layer4.2.conv3",
                name="layer4.2.conv3",
                description="Final bottleneck residual block convolutional layer (Standard Grad-CAM target)",
                isVerified=True,
                depthDescription="Depth: Bottleneck 4.2 · Resolution: 8 × 8",
            ),
        ],
        outputClasses=["Glioma", "Meningioma", "Pituitary", "No Tumor"],
        parametersCount="23.5M",
    ),
]

SEGMENTATION_MODELS: List[SegmentationModelInfo] = [
    SegmentationModelInfo(
        id="brats_3d_unet",
        name="Attention CNN + Vision Transformer (BraTS Multimodal)",
        architecture="Hybrid Attention CNN + Vision Transformer with Additive Attention Gates",
        description="Four-modality volumetric 3D segmentation utilizing aligned FLAIR, T1, T1CE, and T2 sequences. Delineates Necrotic Core, Peritumoral Edema, and Enhancing Tumor.",
        task="Multimodal 3D Semantic Segmentation",
        version="v2.4.0 (BraTS-2020)",
        status="available",
        metrics=SegmentationMetrics(
            diceCoefficient=0.8298,
            iouScore=0.7283,
            sensitivity=0.841,
            inferenceTimeMs=182,
            evaluationDataset="BraTS 2020 Multimodal 56 Test Patient Cohort (Complete Volumes)",
        ),
        classes=["Background", "Necrotic / non-enhancing tumor", "Peritumoral edema", "Enhancing tumor"],
        parametersCount="3.07M",
        supportedXaiMethods=["gradcam", "modality_ablation"],
        compatibleLayers=[
            FeatureLayerOption(
                id="attention_gate_2",
                name="attention_gate_2",
                description="Deepest attention gate isolating active enhancing tumor boundaries",
                isVerified=True,
                depthDescription="Decoder Level 2 · Multi-Head Attention",
            ),
        ],
    ),
    SegmentationModelInfo(
        id="unet3d_brats",
        name="3D U-Net (volumetric)",
        architecture="3D Encoder-Decoder Convolutional Network (Conv3D + BatchNorm)",
        description="Native volumetric sliding-window segmentation on four multimodal BraTS sequences (FLAIR, T1, T1CE, T2) at 96×96×64 patch size.",
        task="Multimodal 3D Semantic Segmentation",
        version="BraTS-3D-UNet-v1",
        status="available",
        metrics=SegmentationMetrics(
            diceCoefficient=0.8298,
            iouScore=0.7283,
            sensitivity=0.841,
            inferenceTimeMs=1850,
            evaluationDataset="BraTS 2020, complete-volume sliding-window evaluation on 56 held-out patients",
        ),
        classes=["Background", "Necrotic / non-enhancing tumor", "Peritumoral edema", "Enhancing tumor"],
        parametersCount="5.61M",
        supportedXaiMethods=["gradcam", "modality_ablation"],
        compatibleLayers=[
            FeatureLayerOption(
                id="bottleneck_relu2",
                name="bottleneck_relu2",
                description="Deepest 3D bottleneck feature activation (12 × 12 × 8 × 256)",
                isVerified=True,
                depthDescription="Bottleneck Level · Resolution: 12 × 12 × 8 × 256",
            ),
        ],
    ),
    SegmentationModelInfo(
        id="deeplabv3plus",
        name="DeepLabV3+",
        architecture="DeepLabV3+ with ResNet encoder and atrous spatial pyramid pooling",
        description="Binary MRI lesion segmentation model with a 256 × 256 RGB input and sigmoid output.",
        task="Binary Lesion Segmentation",
        version="DeepLabV3Plus-256",
        status="available",
        metrics=SegmentationMetrics(
            diceCoefficient=0.848,
            iouScore=0.754,
            sensitivity=0.865,
            inferenceTimeMs=45,
            evaluationDataset="LGG MRI segmentation dataset, patient-grouped split",
        ),
        classes=["Tumor Lesion"],
        parametersCount="17.8M",
        supportedXaiMethods=["seg_gradcam"],
        compatibleLayers=[
            FeatureLayerOption(
                id="conv2d_20",
                name="conv2d_20",
                description="ASPP fused output activation (16 × 16 × 256)",
                isVerified=True,
                depthDescription="ASPP Fused Output · Resolution: 16 × 16 × 256",
            ),
            FeatureLayerOption(
                id="up_sampling2d_2",
                name="up_sampling2d_2",
                description="Last decoder activation before final x4 upsampling (64 × 64 × 256)",
                isVerified=True,
                depthDescription="Decoder Stage · Resolution: 64 × 64 × 256",
            ),
        ],
    ),
    SegmentationModelInfo(
        id="attention_vit_segmentation",
        name="2D Attention CNN + ViT (BraTS Multimodal)",
        architecture="Hybrid Attention CNN + Vision Transformer with Additive Attention Gates",
        description="Four-modality 2D slice segmentation utilizing aligned FLAIR, T1, T1CE, and T2 sequences. Delineates Necrotic Core, Peritumoral Edema, and Enhancing Tumor.",
        task="Multimodal 2D Semantic Segmentation",
        version="v2.4.0 (BraTS-2D)",
        status="available",
        metrics=SegmentationMetrics(
            diceCoefficient=0.8298,
            iouScore=0.7283,
            sensitivity=0.841,
            inferenceTimeMs=85,
            evaluationDataset="BraTS 2020 Multimodal Test Cohort",
        ),
        classes=["Background", "Necrotic / non-enhancing tumor", "Peritumoral edema", "Enhancing tumor"],
        parametersCount="3.07M",
        supportedXaiMethods=["gradcam", "modality_ablation"],
        compatibleLayers=[
            FeatureLayerOption(
                id="attention_gate_2",
                name="attention_gate_2",
                description="Deepest attention gate isolating active enhancing tumor boundaries",
                isVerified=True,
                depthDescription="Decoder Level 2 · Multi-Head Attention",
            ),
        ],
    ),
    SegmentationModelInfo(
        id="unet",
        name="BRISC Binary U-Net",
        architecture="Encoder-Decoder with Symmetric Skip Connections",
        description="A convolutional encoder-decoder architecture structured for binary lesion segmentation (128×128 resolution, single output channel, threshold 0.5).",
        task="Binary Lesion Segmentation",
        version="BRISC-Binary-UNet-Best",
        status="available",
        metrics=SegmentationMetrics(
            diceCoefficient=0.884,
            iouScore=0.792,
            sensitivity=0.891,
            inferenceTimeMs=42,
            evaluationDataset="BRISC 2025 Test Split",
        ),
        classes=["Tumor Lesion"],
        parametersCount="7.86M",
        supportedXaiMethods=["seg_gradcam"],
        compatibleLayers=[
            FeatureLayerOption(
                id="bottleneck.conv2",
                name="bottleneck.conv2",
                description="Deepest latent feature representation capturing global spatial context and tissue contrast",
                isVerified=True,
                depthDescription="Depth: Bottleneck Latent Layer · 512 Channels · 16 × 16",
            ),
            FeatureLayerOption(
                id="decoder.upconv3",
                name="decoder.upconv3",
                description="Expanding decoder stage resolving fine boundary gradients around tumor margins",
                isVerified=True,
                depthDescription="Depth: Decoder Stage 3 · 128 Channels · 64 × 64",
            ),
        ],
    ),
]

XAI_METHODS: List[XaiMethodInfo] = [
    XaiMethodInfo(
        id="gradcam",
        name="Grad-CAM",
        description="Highlights image regions that influence the selected classification output using gradients flowing into the final convolutional feature maps.",
        task="classification",
        status="available",
        supportsLayerSelection=True,
        supportedModels=["efficientnetv2b0_256", "efficientnetv2b0", "swin_transformer", "brats_3d_unet", "unet3d_brats"],
        type="gradient_activation",
    ),
    XaiMethodInfo(
        id="gradcam_pp",
        name="Grad-CAM++",
        description="A gradient-based class activation mapping method that utilizes weighted positive higher-order partial derivatives to provide finer spatial explanations.",
        task="classification",
        status="available",
        supportsLayerSelection=True,
        supportedModels=["efficientnetv2b0_256", "efficientnetv2b0"],
        type="gradient_activation",
    ),
    XaiMethodInfo(
        id="integrated_gradients",
        name="Integrated Gradients",
        description="Attributes a model output to input pixels by accumulating gradients along a linear path between a neutral baseline image and the input scan.",
        task="classification",
        status="available",
        supportsLayerSelection=False,
        supportedModels=["efficientnetv2b0_256", "efficientnetv2b0"],
        type="path_integral",
    ),
    XaiMethodInfo(
        id="seg_gradcam",
        name="Mask-Targeted Grad-CAM",
        description="Computes backpropagated gradients targeting the summed logit activations of the predicted lesion mask to reveal which latent feature channels drove boundary delineation.",
        task="segmentation",
        status="available",
        supportsLayerSelection=True,
        supportedModels=["unet", "deeplabv3plus", "brats_3d_unet", "unet3d_brats"],
        type="gradient_activation",
    ),
    XaiMethodInfo(
        id="mask_guided_cam",
        name="Mask-Guided CAM",
        description="Isolates activation maps spatially conditioned on the segmented Tumor Core ROI, explaining internal heterogeneity within the lesion.",
        task="segmentation",
        status="available",
        supportsLayerSelection=True,
        supportedModels=["unet"],
        type="gradient_activation",
    ),
    XaiMethodInfo(
        id="seg_integrated_gradients",
        name="Segmentation Integrated Gradients",
        description="Calculates path-integral pixel attributions directly conditioned on the segmentation mask loss against a blank reference baseline.",
        task="segmentation",
        status="available",
        supportsLayerSelection=False,
        supportedModels=["unet"],
        type="path_integral",
    ),
]


def get_all_classification_models() -> List[ClassificationModelInfo]:
    status = model_loader.get_status().get("classification_models", {})
    models = []
    for model in CLASSIFICATION_MODELS:
        is_loaded = status.get(model.id, {}).get("available", False)
        details = status.get(model.id, {}).get("details", {})
        models.append(model.model_copy(update={
            "status": "available" if is_loaded else "coming_soon",
            "statusReason": None if is_loaded else (
                details.get("error") or details.get("reason")
                or "Trained checkpoint is not installed on host."
            ),
        }))
    return models


def get_all_segmentation_models() -> List[SegmentationModelInfo]:
    status = model_loader.get_status().get("segmentation_models", {})
    models = []
    for model in SEGMENTATION_MODELS:
        is_loaded = status.get(model.id, {}).get("available", False)
        details = status.get(model.id, {}).get("details", {})
        models.append(model.model_copy(update={
            "status": "available" if is_loaded else "coming_soon",
            "statusReason": None if is_loaded else (
                details.get("error") or details.get("reason")
                or "The trained checkpoint has not been loaded and inference-validated."
            ),
        }))
    return models


def get_all_xai_methods() -> List[XaiMethodInfo]:
    return XAI_METHODS


def get_model_by_id(model_id: str) -> Optional[Any]:
    for m in CLASSIFICATION_MODELS:
        if m.id == model_id:
            return m
    for m in SEGMENTATION_MODELS:
        if m.id == model_id:
            return m
    return None
