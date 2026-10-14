export type AnalysisStatus = 'idle' | 'uploading' | 'preprocessing' | 'classifying' | 'explaining' | 'segmenting' | 'completed' | 'failed';

export interface ClassificationScore {
  label: string;
  score: number;
  isTopPrediction: boolean;
}

export interface ExplanationResult {
  method: 'gradcam' | 'gradcam_pp';
  heatmapDataUrl: string;
  overlayDataUrl: string;
  targetClass: string;
  layerName: string;
  description: string;
}

export interface SegmentationResult {
  maskDataUrl: string;
  overlayDataUrl: string;
  classes: SegmentationClass[];
  description: string;
}

export interface SegmentationClass {
  label: string;
  color: string;
  pixelCount: number;
  percentage: number;
}

export interface AnalysisMetadata {
  modelName: string;
  modelVersion: string;
  processingTimeMs: number;
  imageSize: { width: number; height: number };
  analysisDate: string;
}

export interface AnalysisResult {
  id: string;
  fileName: string;
  imageDataUrl: string;
  status: AnalysisStatus;
  classification: ClassificationScore[] | null;
  predictedClass: string | null;
  confidence: number | null;
  explanations: ExplanationResult[];
  segmentation: SegmentationResult | null;
  metadata: AnalysisMetadata | null;
  isDemo: boolean;
  createdAt: string;
}

export interface AnalysisSummary {
  id: string;
  fileName: string;
  thumbnailUrl: string;
  predictedClass: string | null;
  confidence: number | null;
  status: AnalysisStatus;
  modelVersion: string;
  createdAt: string;
  isDemo: boolean;
}
