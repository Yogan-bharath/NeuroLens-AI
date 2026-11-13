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

export interface ActualModelMetadata {
  model_id: string;
  model_name: string;
  model_version: string;
  architecture: string;
  framework?: string;
  task_type: string;
  input_resolution?: string;
  input_modalities?: string[];
  input_dimensions?: number[];
  voxel_spacing?: number[];
  preprocessing?: string;
  parameters_count?: string;
  inference_time_ms?: number;
  inference_timestamp?: string;
  is_mock?: boolean;
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
  workflow_type?: string;
  analysis_mode?: string;
  model_id?: string;
  model_name?: string;
  model_version?: string;
  actual_model?: ActualModelMetadata;
  volume_dimensions?: [number, number, number];
  voxel_spacing?: [number, number, number];
  anatomical_orientation?: string[];
  class_distribution?: any[];
  slices?: any;
  surface_3d?: any;
  download_url?: string;
  top_prediction?: string;
  scores?: any[];
  gradcam_overlay?: string | null;
  gradcam_pp_overlay?: string | null;
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
  workflow_type?: string;
  analysis_mode?: string;
  model_id?: string;
  model_name?: string;
  actual_model?: ActualModelMetadata;
  gradcam_overlay?: string | null;
  gradcam_pp_overlay?: string | null;
}
