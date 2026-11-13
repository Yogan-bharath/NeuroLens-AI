export type AnalysisMode = 'classification' | 'segmentation' | 'combined';

export type ModelStatus = 'available' | 'coming_soon' | 'unavailable';

export interface ClassificationMetrics {
  testAccuracy?: number;
  macroPrecision?: number;
  macroRecall?: number;
  macroF1?: number;
  inferenceTimeMs?: number;
  evaluationDataset?: string;
}

export interface FeatureLayerOption {
  id: string;
  name: string;
  description: string;
  isVerified: boolean;
  depthDescription?: string;
}

export interface ClassificationModel {
  id: string;
  name: string;
  architecture: string;
  description: string;
  task: 'classification';
  version: string;
  status: ModelStatus;
  statusReason?: string;
  metrics?: ClassificationMetrics;
  supportedXaiMethods: string[];
  compatibleLayers: FeatureLayerOption[];
  outputClasses: string[];
  parametersCount?: string;
}

export interface SegmentationMetrics {
  diceCoefficient?: number;
  iouScore?: number;
  sensitivity?: number;
  inferenceTimeMs?: number;
  evaluationDataset?: string;
}

export interface SegmentationModel {
  id: string;
  name: string;
  architecture: string;
  description: string;
  task: string;
  version?: string;
  status: ModelStatus;
  statusReason?: string;
  metrics?: SegmentationMetrics;
  classes: string[];
  parametersCount?: string;
  supportedXaiMethods: string[];
  compatibleLayers: FeatureLayerOption[];
}

export interface XaiMethod {
  id: string;
  name: string;
  description: string;
  task: 'classification' | 'segmentation' | 'both';
  status: ModelStatus;
  statusReason?: string;
  supportsLayerSelection: boolean;
  supportedModels: string[];
  type: 'gradient_activation' | 'path_integral' | 'mask_saliency';
}

export interface AnalysisConfiguration {
  mode: AnalysisMode;
  classificationModelId: string;
  segmentationModelId: string;
  classificationXaiMethod: string;
  classificationTarget: string;
  classificationLayer: string;
  segmentationXaiMethod: string;
  segmentationLayer: string;
  segmentationVisualization: 'mask' | 'overlay' | 'comparison';
  segmentationOpacity: number;
}
