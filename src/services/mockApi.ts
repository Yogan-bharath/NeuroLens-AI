import type { AnalysisResult, AnalysisSummary, ExplanationResult, SegmentationResult } from '@/types/analysis';
import type { ModelMetrics } from '@/types/model';
import { delay, generateId } from '@/lib/utils';

// ─── Heatmap generation helpers ───────────────────────────────────────────
function generateHeatmapCanvas(width: number, height: number, seed: number): string {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  const imageData = ctx.createImageData(width, height);
  const cx = width * (0.35 + (seed * 0.13) % 0.3);
  const cy = height * (0.35 + (seed * 0.17) % 0.3);
  const maxR = Math.min(width, height) * 0.35;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      let intensity = Math.max(0, 1 - dist / maxR);
      intensity = Math.pow(intensity, 1.5);
      const noise = (Math.sin(x * 0.05 + seed) * Math.cos(y * 0.07 + seed) + 1) * 0.1;
      intensity = Math.min(1, intensity + noise * intensity);
      const i = (y * width + x) * 4;
      // Jet colormap
      if (intensity < 0.25) {
        imageData.data[i] = 0;
        imageData.data[i + 1] = Math.floor(intensity * 4 * 255);
        imageData.data[i + 2] = 255;
      } else if (intensity < 0.5) {
        imageData.data[i] = 0;
        imageData.data[i + 1] = 255;
        imageData.data[i + 2] = Math.floor((1 - (intensity - 0.25) * 4) * 255);
      } else if (intensity < 0.75) {
        imageData.data[i] = Math.floor((intensity - 0.5) * 4 * 255);
        imageData.data[i + 1] = 255;
        imageData.data[i + 2] = 0;
      } else {
        imageData.data[i] = 255;
        imageData.data[i + 1] = Math.floor((1 - (intensity - 0.75) * 4) * 255);
        imageData.data[i + 2] = 0;
      }
      imageData.data[i + 3] = Math.floor(intensity * 200);
    }
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas.toDataURL();
}

function generateSegmentationMask(width: number, height: number): string {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d')!;
  const imageData = ctx.createImageData(width, height);
  const cx = width * 0.45;
  const cy = height * 0.42;
  const rx = width * 0.15;
  const ry = height * 0.18;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const dx = (x - cx) / rx;
      const dy = (y - cy) / ry;
      const dist = dx * dx + dy * dy;
      const i = (y * width + x) * 4;
      if (dist < 1) {
        // Tumor region — teal
        imageData.data[i] = 20;
        imageData.data[i + 1] = 184;
        imageData.data[i + 2] = 166;
        imageData.data[i + 3] = 160;
      } else if (dist < 1.6) {
        // Edema — yellow
        imageData.data[i] = 245;
        imageData.data[i + 1] = 158;
        imageData.data[i + 2] = 11;
        imageData.data[i + 3] = 100;
      } else {
        imageData.data[i + 3] = 0;
      }
    }
  }
  ctx.putImageData(imageData, 0, 0);
  return canvas.toDataURL();
}

// ─── Demo placeholder image ──────────────────────────────────────────────
function createDemoMRIPlaceholder(label: string): string {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext('2d')!;
  // Dark background
  ctx.fillStyle = '#1a1a2e';
  ctx.fillRect(0, 0, 256, 256);
  // Simulated brain outline
  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.ellipse(128, 128, 90, 100, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.ellipse(128, 128, 70, 80, 0, 0, Math.PI * 2);
  ctx.stroke();
  // Inner structures
  ctx.strokeStyle = '#475569';
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  ctx.moveTo(128, 40);
  ctx.lineTo(128, 220);
  ctx.stroke();
  ctx.beginPath();
  ctx.ellipse(105, 110, 25, 20, -0.2, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.ellipse(151, 110, 25, 20, 0.2, 0, Math.PI * 2);
  ctx.stroke();
  // Label
  ctx.fillStyle = '#64748b';
  ctx.font = '11px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(label, 128, 245);
  ctx.fillStyle = '#475569';
  ctx.font = '9px Inter, sans-serif';
  ctx.fillText('Demo MRI Placeholder', 128, 20);
  return canvas.toDataURL();
}

// ─── Demo data ───────────────────────────────────────────────────────────
const demoImages = {
  'MRI-001': createDemoMRIPlaceholder('MRI-001'),
  'MRI-002': createDemoMRIPlaceholder('MRI-002'),
  'MRI-003': createDemoMRIPlaceholder('MRI-003'),
};

const demoAnalyses: AnalysisResult[] = [
  {
    id: 'demo-001',
    fileName: 'brain_mri_demo_001.png',
    imageDataUrl: demoImages['MRI-001'],
    status: 'completed',
    classification: [
      { label: 'Glioma', score: 0.847, isTopPrediction: true },
      { label: 'Meningioma', score: 0.098, isTopPrediction: false },
      { label: 'Pituitary', score: 0.041, isTopPrediction: false },
      { label: 'No Tumor', score: 0.014, isTopPrediction: false },
    ],
    predictedClass: 'Glioma',
    confidence: 0.847,
    explanations: [],
    segmentation: null,
    metadata: {
      modelName: 'NeuroClassifier',
      modelVersion: 'v2.1.0',
      processingTimeMs: 1243,
      imageSize: { width: 256, height: 256 },
      analysisDate: '2024-10-12T14:30:00Z',
    },
    isDemo: true,
    createdAt: '2024-10-12T14:30:00Z',
  },
  {
    id: 'demo-002',
    fileName: 'brain_mri_demo_002.png',
    imageDataUrl: demoImages['MRI-002'],
    status: 'completed',
    classification: [
      { label: 'Glioma', score: 0.052, isTopPrediction: false },
      { label: 'Meningioma', score: 0.891, isTopPrediction: true },
      { label: 'Pituitary', score: 0.033, isTopPrediction: false },
      { label: 'No Tumor', score: 0.024, isTopPrediction: false },
    ],
    predictedClass: 'Meningioma',
    confidence: 0.891,
    explanations: [],
    segmentation: null,
    metadata: {
      modelName: 'NeuroClassifier',
      modelVersion: 'v2.1.0',
      processingTimeMs: 1102,
      imageSize: { width: 256, height: 256 },
      analysisDate: '2024-10-11T09:15:00Z',
    },
    isDemo: true,
    createdAt: '2024-10-11T09:15:00Z',
  },
  {
    id: 'demo-003',
    fileName: 'brain_mri_demo_003.png',
    imageDataUrl: demoImages['MRI-003'],
    status: 'completed',
    classification: [
      { label: 'Glioma', score: 0.021, isTopPrediction: false },
      { label: 'Meningioma', score: 0.015, isTopPrediction: false },
      { label: 'Pituitary', score: 0.032, isTopPrediction: false },
      { label: 'No Tumor', score: 0.932, isTopPrediction: true },
    ],
    predictedClass: 'No Tumor',
    confidence: 0.932,
    explanations: [],
    segmentation: null,
    metadata: {
      modelName: 'NeuroClassifier',
      modelVersion: 'v2.1.0',
      processingTimeMs: 987,
      imageSize: { width: 256, height: 256 },
      analysisDate: '2024-10-10T16:45:00Z',
    },
    isDemo: true,
    createdAt: '2024-10-10T16:45:00Z',
  },
];

// ─── Mock model metrics ──────────────────────────────────────────────────
const demoModelMetrics: ModelMetrics = {
  accuracy: 0.923,
  macroPrecision: 0.918,
  macroRecall: 0.912,
  macroF1: 0.915,
  perClassMetrics: [
    { className: 'Glioma', precision: 0.931, recall: 0.918, f1Score: 0.924, support: 300 },
    { className: 'Meningioma', precision: 0.907, recall: 0.895, f1Score: 0.901, support: 306 },
    { className: 'Pituitary', precision: 0.942, recall: 0.936, f1Score: 0.939, support: 300 },
    { className: 'No Tumor', precision: 0.893, recall: 0.899, f1Score: 0.896, support: 405 },
  ],
  confusionMatrix: {
    labels: ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor'],
    matrix: [
      [275, 12, 5, 8],
      [10, 274, 8, 14],
      [4, 6, 281, 9],
      [7, 18, 16, 364],
    ],
  },
  metadata: {
    name: 'NeuroClassifier',
    version: 'v2.1.0',
    architecture: 'ResNet-50 (Modified)',
    framework: 'PyTorch 2.1',
    inputSize: '256 × 256 × 1',
    totalParameters: '23.5M',
    trainingDataset: 'Example Brain Tumor MRI Dataset',
    evaluationSplit: 'Test set (20%)',
    lastUpdated: '2024-09-15',
  },
  isExampleData: true,
};

// ─── Service functions ───────────────────────────────────────────────────
import type { AnalysisConfiguration } from '@/types/modelRegistry';

export async function mockAnalyzeMRI(
  imageDataUrl: string,
  fileName: string,
  config?: AnalysisConfiguration
): Promise<AnalysisResult> {
  // Stage 1: Upload
  await delay(600);
  // Stage 2: Preprocessing
  await delay(800);
  // Stage 3: Classification
  await delay(1200);
  // Stage 4: Explanations
  await delay(1000);

  const gradcamHeatmap = generateHeatmapCanvas(256, 256, 1);
  const gradcamPPHeatmap = generateHeatmapCanvas(256, 256, 2);
  const segMask = generateSegmentationMask(256, 256);

  const targetClass = 'Glioma';
  const layerName = config?.classificationLayer || (config?.classificationModelId === 'efficientnetv2b0' ? 'top_conv' : 'layer4.2.conv3');

  const explanations: ExplanationResult[] = [
    {
      method: 'gradcam',
      heatmapDataUrl: gradcamHeatmap,
      overlayDataUrl: gradcamHeatmap,
      targetClass,
      layerName,
      description: `Gradient-weighted Class Activation Map extracted from ${layerName} highlighting regions that most influenced the ${targetClass} decision.`,
    },
    {
      method: 'gradcam_pp',
      heatmapDataUrl: gradcamPPHeatmap,
      overlayDataUrl: gradcamPPHeatmap,
      targetClass,
      layerName,
      description: `Grad-CAM++ higher-order saliency extracted from ${layerName} providing multi-focal spatial attribution.`,
    },
  ];

  const segLayer = config?.segmentationLayer || 'bottleneck.conv2';
  const segXaiMethod = config?.segmentationXaiMethod || 'seg_gradcam';

  const segmentation: SegmentationResult = {
    maskDataUrl: segMask,
    overlayDataUrl: segMask,
    classes: [
      { label: 'Tumor Core', color: '#14b8a6', pixelCount: 4820, percentage: 7.35 },
      { label: 'Peritumoral Edema', color: '#f59e0b', pixelCount: 8340, percentage: 12.72 },
    ],
    description: `Predicted semantic segmentation mask from U-Net. Boundary gradients verified via ${segXaiMethod} on latent feature layer "${segLayer}".`,
  };

  const isSegOnly = config?.mode === 'segmentation';
  const modelName = config?.classificationModelId === 'efficientnetv2b0'
    ? 'EfficientNetV2B0'
    : config?.classificationModelId === 'densenet121'
    ? 'DenseNet121'
    : 'ResNet50';

  const modelVersion = config?.classificationModelId === 'efficientnetv2b0'
    ? 'v2.4.0'
    : 'v2.1.0';

  return {
    id: generateId(),
    fileName,
    imageDataUrl,
    status: 'completed',
    classification: isSegOnly ? null : [
      { label: 'Glioma', score: 0.847, isTopPrediction: true },
      { label: 'Meningioma', score: 0.098, isTopPrediction: false },
      { label: 'Pituitary', score: 0.041, isTopPrediction: false },
      { label: 'No Tumor', score: 0.014, isTopPrediction: false },
    ],
    predictedClass: isSegOnly ? 'Segmentation Output' : 'Glioma',
    confidence: isSegOnly ? 0.884 : 0.847,
    explanations: isSegOnly ? [] : explanations,
    segmentation: config?.mode === 'classification' ? null : segmentation,
    metadata: {
      modelName: isSegOnly ? 'U-Net' : modelName,
      modelVersion,
      processingTimeMs: 3600,
      imageSize: { width: 256, height: 256 },
      analysisDate: new Date().toISOString(),
    },
    isDemo: true,
    createdAt: new Date().toISOString(),
  };
}

export async function mockGetAnalysis(id: string): Promise<AnalysisResult | null> {
  await delay(300);
  const demo = demoAnalyses.find(a => a.id === id);
  if (demo) {
    // Lazily generate explanations for demo analyses
    if (demo.explanations.length === 0) {
      demo.explanations = [
        {
          method: 'gradcam',
          heatmapDataUrl: generateHeatmapCanvas(256, 256, 1),
          overlayDataUrl: generateHeatmapCanvas(256, 256, 1),
          targetClass: demo.predictedClass || 'Unknown',
          layerName: 'layer4',
          description: 'Gradient-weighted Class Activation Map highlighting regions that most influenced the classification decision.',
        },
        {
          method: 'gradcam_pp',
          heatmapDataUrl: generateHeatmapCanvas(256, 256, 2),
          overlayDataUrl: generateHeatmapCanvas(256, 256, 2),
          targetClass: demo.predictedClass || 'Unknown',
          layerName: 'layer4',
          description: 'Grad-CAM++ provides improved localization by using weighted combinations of positive partial derivatives.',
        },
      ];
      demo.segmentation = {
        maskDataUrl: generateSegmentationMask(256, 256),
        overlayDataUrl: generateSegmentationMask(256, 256),
        classes: [
          { label: 'Tumor Core', color: '#14b8a6', pixelCount: 4820, percentage: 7.35 },
          { label: 'Peritumoral Edema', color: '#f59e0b', pixelCount: 8340, percentage: 12.72 },
        ],
        description: 'Predicted segmentation mask showing tumor region boundaries.',
      };
    }
    return demo;
  }
  return null;
}

export async function mockGetAnalysisHistory(): Promise<AnalysisSummary[]> {
  await delay(400);
  return demoAnalyses.map(a => ({
    id: a.id,
    fileName: a.fileName,
    thumbnailUrl: a.imageDataUrl,
    predictedClass: a.predictedClass,
    confidence: a.confidence,
    status: a.status,
    modelVersion: a.metadata?.modelVersion || 'Unknown',
    createdAt: a.createdAt,
    isDemo: a.isDemo,
  }));
}

export async function mockGetModelMetrics(): Promise<ModelMetrics> {
  await delay(500);
  return demoModelMetrics;
}

export function getDemoAnalyses(): AnalysisResult[] {
  return demoAnalyses;
}

export function addAnalysisToHistory(analysis: AnalysisResult): void {
  demoAnalyses.unshift(analysis);
}
