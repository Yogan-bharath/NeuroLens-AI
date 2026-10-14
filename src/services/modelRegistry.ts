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
    id: 'densenet121',
    name: 'DenseNet121',
    architecture: 'Densely Connected Convolutional Network 121',
    description: 'A densely connected convolutional architecture that connects each layer to every other layer in a feed-forward fashion.',
    task: 'classification',
    version: 'v1.1.0-rc',
    status: 'coming_soon',
    statusReason: 'Model checkpoint undergoing independent test split benchmark validation. Integration pending.',
    metrics: undefined, // Unmeasured metrics displayed as "Not evaluated"
    supportedXaiMethods: ['gradcam', 'integrated_gradients'], // Does NOT support Grad-CAM++ in this build
    compatibleLayers: [
      {
        id: 'features.denseblock4',
        name: 'features.denseblock4',
        description: 'Final dense block with dense concatenated feature channels',
        isVerified: true,
        depthDescription: 'Depth: Final Dense Block · Resolution: 8 × 8',
      },
      {
        id: 'features.transition3',
        name: 'features.transition3',
        description: 'Transition layer 3 prior to deep feature convergence',
        isVerified: true,
        depthDescription: 'Depth: Intermediate Transition · Resolution: 16 × 16',
      },
    ],
    outputClasses: ['Glioma', 'Meningioma', 'Pituitary tumor', 'No tumor detected by the model'],
    parametersCount: '8.0M',
  },
  {
    id: 'resnet50',
    name: 'ResNet50',
    architecture: 'Residual Neural Network 50-Layer',
    description: 'A deep residual neural network architecture utilizing identity shortcut connections to mitigate vanishing gradient degradation.',
    task: 'classification',
    version: 'v2.1.0',
    status: 'available',
    metrics: {
      testAccuracy: 0.923,
      macroPrecision: 0.918,
      macroRecall: 0.912,
      macroF1: 0.915,
      inferenceTimeMs: 42,
      evaluationDataset: 'Standard Brain Tumor MRI Test Split (Held-out)',
    },
    supportedXaiMethods: ['gradcam', 'gradcam_pp', 'integrated_gradients'],
    compatibleLayers: [
      {
        id: 'layer4.2.conv3',
        name: 'layer4.2.conv3',
        description: 'Final bottleneck residual block convolutional layer (Standard Grad-CAM target)',
        isVerified: true,
        depthDescription: 'Depth: Bottleneck 4.2 · Resolution: 8 × 8',
      },
      {
        id: 'layer4.1.conv2',
        name: 'layer4.1.conv2',
        description: 'Intermediate bottleneck layer 4 block 1 3×3 convolution',
        isVerified: true,
        depthDescription: 'Depth: Bottleneck 4.1 · Resolution: 8 × 8',
      },
      {
        id: 'layer3.5.conv3',
        name: 'layer3.5.conv3',
        description: 'Stage 3 deep bottleneck feature maps prior to final downsampling',
        isVerified: false,
        depthDescription: 'Depth: Bottleneck 3.5 · Resolution: 16 × 16',
      },
    ],
    outputClasses: ['Glioma', 'Meningioma', 'Pituitary tumor', 'No tumor detected by the model'],
    parametersCount: '23.5M',
  },
];

// ─── Segmentation Models Registry ─────────────────────────────────────────
export const segmentationModelsRegistry: SegmentationModel[] = [
  {
    id: 'unet',
    name: 'U-Net',
    architecture: 'Encoder-Decoder with Symmetric Skip Connections',
    description: 'An encoder-decoder architecture specifically structured for biomedical pixel-level segmentation, enabling precise localization via contracting and expanding paths.',
    task: 'Multiclass Semantic Segmentation',
    version: 'v1.4.2',
    status: 'available',
    metrics: {
      diceCoefficient: 0.884,
      iouScore: 0.792,
      sensitivity: 0.891,
      inferenceTimeMs: 64,
      evaluationDataset: 'BraTS Multimodal Validation Split',
    },
    classes: ['Tumor Core', 'Peritumoral Edema', 'Healthy Tissue'],
    parametersCount: '31.0M',
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
      {
        id: 'encoder.conv4',
        name: 'encoder.conv4',
        description: 'Contracting encoder stage extracting structural boundary texture',
        isVerified: false,
        depthDescription: 'Depth: Encoder Stage 4 · 256 Channels · 32 × 32',
      },
    ],
  },
  {
    id: 'attention_unet',
    name: 'Attention U-Net',
    architecture: 'U-Net with Additive Attention Gates',
    description: 'An attention-enhanced U-Net architecture intended to emphasize relevant spatial features while suppressing irrelevant activations along skip connections.',
    task: 'Multiclass Semantic Segmentation',
    version: 'v0.9.0-alpha',
    status: 'coming_soon',
    statusReason: 'Attention gate weights are currently undergoing training pipeline evaluation; model integration is pending.',
    metrics: undefined, // Not evaluated
    classes: ['Tumor Core', 'Peritumoral Edema'],
    parametersCount: '34.8M',
    supportedXaiMethods: ['seg_gradcam', 'mask_guided_cam'],
    compatibleLayers: [
      {
        id: 'attention_gate.stage4',
        name: 'attention_gate.stage4',
        description: 'Attention gating spatial filtering coefficients',
        isVerified: false,
        depthDescription: 'Depth: Attention Gate 4 · 256 Channels',
      },
    ],
  },
  {
    id: 'unet_plus_plus',
    name: 'U-Net++',
    architecture: 'Nested Dense Skip-Connected U-Net',
    description: 'A nested U-Net architecture with redesigned skip connections and deep supervision to bridge semantic gap between encoder and decoder sub-networks.',
    task: 'Multiclass Semantic Segmentation',
    version: 'v0.5.0-alpha',
    status: 'coming_soon',
    statusReason: 'Nested dense skip-connection implementation pending validation on clinical test cohort.',
    metrics: undefined, // Not evaluated
    classes: ['Tumor Core', 'Peritumoral Edema'],
    parametersCount: '36.6M',
    supportedXaiMethods: ['seg_gradcam'],
    compatibleLayers: [
      {
        id: 'nested_dense_block.0_3',
        name: 'nested_dense_block.0_3',
        description: 'Nested dense skip node feature aggregation',
        isVerified: false,
        depthDescription: 'Depth: Dense Node X_0,3',
      },
    ],
  },
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
    supportedModels: ['efficientnetv2b0', 'densenet121', 'resnet50'],
    type: 'gradient_activation',
  },
  {
    id: 'gradcam_pp',
    name: 'Grad-CAM++',
    description: 'A gradient-based class activation mapping method that utilizes weighted positive higher-order partial derivatives to provide finer spatial explanations.',
    task: 'classification',
    status: 'available',
    supportsLayerSelection: true,
    supportedModels: ['efficientnetv2b0', 'resnet50'],
    type: 'gradient_activation',
  },
  {
    id: 'integrated_gradients',
    name: 'Integrated Gradients',
    description: 'Attributes a model output to input pixels by accumulating gradients along a linear path between a neutral baseline image and the input scan.',
    task: 'classification',
    status: 'available',
    supportsLayerSelection: false,
    supportedModels: ['efficientnetv2b0', 'densenet121', 'resnet50'],
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
    supportedModels: ['unet', 'attention_unet', 'unet_plus_plus'],
    type: 'gradient_activation',
  },
  {
    id: 'mask_guided_cam',
    name: 'Mask-Guided CAM',
    description: 'Isolates activation maps spatially conditioned on the segmented Tumor Core ROI, explaining internal heterogeneity within the lesion.',
    task: 'segmentation',
    status: 'available',
    supportsLayerSelection: true,
    supportedModels: ['unet', 'attention_unet'],
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
  return classificationModelsRegistry;
}

export async function getSegmentationModels(): Promise<SegmentationModel[]> {
  return segmentationModelsRegistry;
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

export function validateConfiguration(config: AnalysisConfiguration): ValidationResult {
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
    const segModel = segmentationModelsRegistry.find(m => m.id === config.segmentationModelId);
    if (!segModel) {
      errors.push('A valid segmentation model must be selected.');
    } else if (segModel.status !== 'available') {
      errors.push(`Selected segmentation model "${segModel.name}" is not currently available (${segModel.statusReason || 'Coming Soon'}).`);
    } else {
      const segXai = xaiMethodsRegistry.find(x => x.id === config.segmentationXaiMethod);
      if (segXai && !segXai.supportedModels.includes(segModel.id)) {
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
