import type {
  ClassificationModel,
  SegmentationModel,
  XaiMethod,
  AnalysisConfiguration,
  FeatureLayerOption,
} from '@/types/modelRegistry';

// ─── Classification Models Registry ───────────────────────────────────────
export const classificationModelsRegistry: ClassificationModel[] = [
  {
    id: 'efficientnetv2b0',
    name: 'EfficientNetV2B0',
    architecture: 'EfficientNetV2-B0 (Fused-MBConv)',
    description: 'A convolutional neural network optimized for four-class brain MRI classification with fast convergence and low parameter footprint.',
    task: 'classification',
    version: 'v2.4.0',
    status: 'available',
    metrics: {
      testAccuracy: 0.941,
      macroPrecision: 0.938,
      macroRecall: 0.935,
      macroF1: 0.936,
      inferenceTimeMs: 38,
      evaluationDataset: 'BraTS / Figshare Benchmark Test Split',
    },
    supportedXaiMethods: ['gradcam', 'gradcam_pp', 'integrated_gradients'],
    compatibleLayers: [
      {
        id: 'top_conv',
        name: 'top_conv',
        description: 'Final 1×1 convolutional projection layer prior to global pooling',
        isVerified: true,
        depthDescription: 'Depth: Layer 168 · Resolution: 8 × 8',
      },
      {
        id: 'block4b_expand_activation',
        name: 'block4b_expand_activation',
        description: 'Intermediate stage expand activation capturing regional mid-level patterns',
        isVerified: true,
        depthDescription: 'Depth: Layer 112 · Resolution: 16 × 16',
      },
      {
        id: 'block3b_expand_activation',
        name: 'block3b_expand_activation',
        description: 'Early-mid layer capturing edge and texture representations',
        isVerified: false,
        depthDescription: 'Depth: Layer 74 · Resolution: 32 × 32',
      },
    ],
    outputClasses: ['Glioma', 'Meningioma', 'Pituitary tumor', 'No tumor detected by the model'],
    parametersCount: '7.1M',
  },
  {
    id: 'swin_transformer',
    name: 'Swin Transformer',
    architecture: 'Hierarchical Shifted-Window Transformer',
    description: 'A trained 224×224 RGB Swin Transformer classifier. Transformer Grad-CAM is generated from its final spatial feature block.',
    task: 'classification',
    version: 'SwinTransformer-224',
    status: 'available',
    metrics: undefined,
    supportedXaiMethods: ['gradcam'],
    compatibleLayers: [{
      id: 'swin_transformer_block_7',
      name: 'swin_transformer_block_7',
      description: 'Final shifted-window spatial feature block before global pooling',
      isVerified: true,
      depthDescription: 'Resolution: 7 × 7',
    }],
    outputClasses: ['Glioma', 'Meningioma', 'Pituitary tumor', 'No tumor detected by the model'],
    parametersCount: 'Artifact-defined',
  },
];

// ─── 3D Volumetric Segmentation Models Registry ───────────────────────────
export const volumetricSegmentationModelsRegistry: SegmentationModel[] = [
  {
    id: 'brats_3d_unet',
    name: 'Attention CNN + Vision Transformer',
    architecture: 'Hybrid Attention CNN + Vision Transformer with Additive Attention Gates',
    description: 'Four-modality volumetric 3D segmentation utilizing aligned FLAIR, T1, T1CE, and T2 sequences. Delineates Necrotic Core, Peritumoral Edema, and Enhancing Tumor.',
    task: 'Multimodal 3D Semantic Segmentation',
    version: 'v2.4.0 (BraTS-2020)',
    status: 'available',
    metrics: {
      diceCoefficient: 0.8298,
      iouScore: 0.7283,
      sensitivity: 0.841,
      inferenceTimeMs: 182,
      evaluationDataset: 'BraTS 2020 Multimodal 56 Test Cohort (Complete Volumes)',
    },
    classes: ['Background', 'Necrotic / non-enhancing tumor', 'Peritumoral edema', 'Enhancing tumor'],
    parametersCount: '3.07M',
    supportedXaiMethods: ['seg_gradcam'],
    compatibleLayers: [
      {
        id: 'attention_gate_2',
        name: 'attention_gate_2',
        description: 'Deepest attention gate isolating active enhancing tumor boundaries',
        isVerified: true,
        depthDescription: 'Decoder Level 2 · Multi-Head Attention',
      },
    ],
  },
];

// ─── 2D Binary Lesion Segmentation Models Registry ─────────────────────────
export const binarySegmentationModelsRegistry: SegmentationModel[] = [
  {
    id: 'unet',
    name: 'BRISC Binary U-Net',
    architecture: 'Encoder-Decoder with Symmetric Skip Connections',
    description: 'An encoder-decoder architecture specifically structured for biomedical pixel-level segmentation, enabling precise localization via contracting and expanding paths.',
    task: 'Binary Lesion Segmentation (2D)',
    version: 'v1.4.2',
    status: 'available',
    metrics: {
      diceCoefficient: 0.884,
      iouScore: 0.792,
      sensitivity: 0.891,
      inferenceTimeMs: 64,
      evaluationDataset: 'BRISC 2025 Test Split',
    },
    classes: ['Tumor Lesion'],
    parametersCount: '7.86M',
    supportedXaiMethods: ['seg_gradcam', 'mask_guided_cam', 'seg_integrated_gradients'],
    compatibleLayers: [
      {
        id: 'bottleneck.conv2',
        name: 'bottleneck.conv2',
        description: 'Deepest latent feature representation capturing global spatial context and tissue contrast',
        isVerified: true,
        depthDescription: 'Depth: Bottleneck Latent Layer · 512 Channels · 16 × 16',
      },
      {
        id: 'decoder.upconv3',
        name: 'decoder.upconv3',
        description: 'Expanding decoder stage resolving fine boundary gradients around tumor margins',
        isVerified: true,
        depthDescription: 'Depth: Decoder Stage 3 · 128 Channels · 64 × 64',
      },
    ],
  },
  {
    id: 'deeplabv3plus',
    name: 'DeepLabV3+',
    architecture: 'DeepLabV3+ with ResNet Encoder and Atrous Spatial Pyramid Pooling',
    description: 'Binary MRI lesion segmentation using a trained 256×256 RGB DeepLabV3+ model.',
    task: 'Binary Lesion Segmentation (2D)',
    version: 'DeepLabV3Plus-256',
    status: 'available',
    metrics: undefined,
    classes: ['Tumor Lesion'],
    parametersCount: '17.8M',
    supportedXaiMethods: ['seg_gradcam'],
    compatibleLayers: [
      {
        id: 'conv2d_26',
        name: 'conv2d_26',
        description: 'Decoder feature map before final sigmoid projection',
        isVerified: true,
        depthDescription: 'DeepLab decoder feature map',
      },
    ],
  },
];

// Combined for backward compatibility across existing views
export const segmentationModelsRegistry: SegmentationModel[] = [
  ...volumetricSegmentationModelsRegistry,
  ...binarySegmentationModelsRegistry,
];

// ─── XAI Methods Registry ────────────────────────────────────────────────
export const xaiMethodsRegistry: XaiMethod[] = [
  // Classification XAI
  {
    id: 'gradcam',
    name: 'Grad-CAM',
    description: 'Highlights image regions that influence the selected classification output using gradients flowing into the final convolutional feature maps.',
    task: 'classification',
    status: 'available',
    supportsLayerSelection: true,
    supportedModels: ['efficientnetv2b0', 'swin_transformer'],
    type: 'gradient_activation',
  },
  {
    id: 'gradcam_pp',
    name: 'Grad-CAM++',
    description: 'A gradient-based class activation mapping method that utilizes weighted positive higher-order partial derivatives to provide finer spatial explanations.',
    task: 'classification',
    status: 'available',
    supportsLayerSelection: true,
    supportedModels: ['efficientnetv2b0'],
    type: 'gradient_activation',
  },
  {
    id: 'integrated_gradients',
    name: 'Integrated Gradients',
    description: 'Attributes a model output to input pixels by accumulating gradients along a linear path between a neutral baseline image and the input scan.',
    task: 'classification',
    status: 'available',
    supportsLayerSelection: false,
    supportedModels: ['efficientnetv2b0', 'swin_transformer'],
    type: 'path_integral',
  },

  // Segmentation XAI
  {
    id: 'seg_gradcam',
    name: 'Mask-Targeted Grad-CAM',
    description: 'Computes backpropagated gradients targeting the summed logit activations of the predicted lesion mask to reveal which latent feature channels drove boundary delineation.',
    task: 'segmentation',
    status: 'available',
    supportsLayerSelection: true,
    supportedModels: ['brats_3d_unet', 'unet', 'deeplabv3plus'],
    type: 'gradient_activation',
  },
  {
    id: 'mask_guided_cam',
    name: 'Mask-Guided CAM',
    description: 'Isolates activation maps spatially conditioned on the segmented Tumor Core ROI, explaining internal heterogeneity within the lesion.',
    task: 'segmentation',
    status: 'available',
    supportsLayerSelection: true,
    supportedModels: ['brats_3d_unet', 'unet'],
    type: 'gradient_activation',
  },
  {
    id: 'seg_integrated_gradients',
    name: 'Segmentation Integrated Gradients',
    description: 'Calculates path-integral pixel attributions directly conditioned on the segmentation mask loss against a blank reference baseline.',
    task: 'segmentation',
    status: 'available',
    supportsLayerSelection: false,
    supportedModels: ['unet'],
    type: 'path_integral',
  },
];

// ─── Service Queries ──────────────────────────────────────────────────────
export async function getClassificationModels(): Promise<ClassificationModel[]> {
  return classificationModelsRegistry.filter(m => m.status === 'available');
}

export function getVolumetricSegmentationModels(): SegmentationModel[] {
  return volumetricSegmentationModelsRegistry.filter(m => m.status === 'available');
}

export function getBinarySegmentationModels(): SegmentationModel[] {
  return binarySegmentationModelsRegistry.filter(m => m.status === 'available');
}

export async function getSegmentationModels(): Promise<SegmentationModel[]> {
  const apiBaseUrl = import.meta.env?.VITE_API_BASE_URL || null;
  if (!apiBaseUrl) return segmentationModelsRegistry.filter(m => m.status === 'available');

  try {
    const response = await fetch(`${apiBaseUrl}/api/models/segmentation`);
    if (!response.ok) {
      return segmentationModelsRegistry.filter(m => m.status === 'available');
    }
    const backendModels: SegmentationModel[] = await response.json();
    const backendById = new Map(backendModels.map(model => [model.id, model]));

    return segmentationModelsRegistry
      .map(model => {
        const backendModel = backendById.get(model.id);
        if (!backendModel) return model;
        return {
          ...model,
          status: backendModel.status,
          statusReason: backendModel.statusReason,
        };
      })
      .filter(model => model.status === 'available');
  } catch {
    return segmentationModelsRegistry.filter(m => m.status === 'available');
  }
}

export async function getXaiMethodsForModel(modelId: string): Promise<XaiMethod[]> {
  return xaiMethodsRegistry.filter(method => method.supportedModels.includes(modelId));
}

export function getCompatibleLayersForModel(modelId: string): FeatureLayerOption[] {
  const model = classificationModelsRegistry.find(m => m.id === modelId);
  return model?.compatibleLayers || [];
}

export function getCompatibleSegmentationLayersForModel(modelId: string): FeatureLayerOption[] {
  const model = segmentationModelsRegistry.find(m => m.id === modelId);
  return model?.compatibleLayers || [];
}

// ─── Validation Service ──────────────────────────────────────────────────
export interface ValidationResult {
  isValid: boolean;
  errors: string[];
  warnings: string[];
}

export function validateConfiguration(
  config: AnalysisConfiguration,
  segmentationModels: SegmentationModel[] = segmentationModelsRegistry
): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  // 1. Classification check
  if (config.mode === 'classification' || config.mode === 'combined') {
    const classModel = classificationModelsRegistry.find(m => m.id === config.classificationModelId);
    if (!classModel) {
      errors.push('A valid classification model must be selected.');
    } else if (classModel.status !== 'available') {
      errors.push(`Selected classification model "${classModel.name}" is not currently available (${classModel.statusReason || 'Coming Soon'}).`);
    } else {
      const xai = xaiMethodsRegistry.find(x => x.id === config.classificationXaiMethod);
      if (!xai) {
        errors.push('An explainable AI (XAI) method must be selected for classification.');
      } else if (!xai.supportedModels.includes(classModel.id)) {
        errors.push(`XAI method "${xai.name}" is not compatible with "${classModel.name}".`);
      }
    }
  }

  // 2. Segmentation check
  if (config.mode === 'segmentation' || config.mode === 'combined') {
    const segModel = segmentationModels.find(m => m.id === config.segmentationModelId);
    if (!segModel) {
      errors.push('A valid segmentation model must be selected.');
    } else if (segModel.status !== 'available') {
      errors.push(`Selected segmentation model "${segModel.name}" is not currently available (${segModel.statusReason || 'Coming Soon'}).`);
    } else {
      if (segModel.id === 'brats_3d_unet' && config.mode !== 'segmentation') {
        errors.push('BraTS 3D U-Net currently supports segmentation-only analysis.');
      }
      const segXai = xaiMethodsRegistry.find(x => x.id === config.segmentationXaiMethod);
      if (segModel.supportedXaiMethods.length > 0 && segXai && !segXai.supportedModels.includes(segModel.id)) {
        errors.push(`Segmentation XAI method "${segXai.name}" is not compatible with "${segModel.name}".`);
      }
    }
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
  };
}
