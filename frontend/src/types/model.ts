export interface ModelMetadata {
  name: string;
  version: string;
  architecture: string;
  framework: string;
  inputSize: string;
  totalParameters: string;
  trainingDataset: string;
  evaluationSplit: string;
  lastUpdated: string;
  evaluationProtocol?: string;
}

export interface PerClassMetrics {
  className: string;
  precision: number;
  recall: number;
  f1Score: number;
  support: number;
}

export interface ConfusionMatrixData {
  labels: string[];
  matrix: number[][];
}

export interface ModelMetrics {
  accuracy: number;
  macroPrecision: number;
  macroRecall: number;
  macroF1: number;
  perClassMetrics: PerClassMetrics[];
  confusionMatrix: ConfusionMatrixData;
  metadata: ModelMetadata;
  isExampleData: boolean;
}

// ─── Classification Performance Model ──────────────────────────────────────
export interface ClassificationModelItem {
  model_id: string;
  name: string;
  version: string;
  architecture: string;
  framework: string;
  input_size: string;
  parameters: string;
  dataset: string;
  evaluation_split: string;
  last_evaluated?: string | null;
  evaluated: boolean;
  status: string; // "Evaluated" | "Not evaluated"
  reason?: string | null;
  accuracy?: number | null;
  macro_precision?: number | null;
  macro_recall?: number | null;
  macro_f1?: number | null;
  specificity?: number | null;
  sensitivity?: number | null;
  average_inference_time_ms?: number | null;
  per_class_metrics?: PerClassMetrics[] | null;
  confusion_matrix?: ConfusionMatrixData | null;
}

export interface ClassificationPerformanceResponse {
  models: ClassificationModelItem[];
  best_model_id: string;
  benchmark_dataset: string;
  total_test_samples: number;
}

// ─── Segmentation Performance Model ────────────────────────────────────────
export interface PerRegionMetrics {
  region_name: string;
  dice_score: number;
  iou_score: number;
}

export interface SegmentationModelItem {
  model_id: string;
  name: string;
  version: string;
  architecture: string;
  framework: string;
  input_size: string;
  parameters: string;
  dataset: string;
  evaluation_protocol: string;
  last_evaluated?: string | null;
  evaluated: boolean;
  status: string; // "Evaluated" | "Not evaluated"
  reason?: string | null;
  dice_coefficient?: number | null;
  mean_iou?: number | null;
  precision?: number | null;
  recall?: number | null;
  sensitivity?: number | null;
  specificity?: number | null;
  pixel_accuracy?: number | null;
  average_inference_time_ms?: number | null;
  per_region_metrics?: PerRegionMetrics[] | null;
  classes: string[];
}

export interface SegmentationPerformanceResponse {
  models: SegmentationModelItem[];
  best_model_id: string;
  evaluation_cohort: string;
  total_test_patients: number;
}

// ─── Unified Model Overview & Comparison ──────────────────────────────────
export interface UnifiedModelOverview {
  model_id: string;
  name: string;
  category: 'classification' | 'segmentation';
  architecture: string;
  parameters: string;
  input_size: string;
  evaluated: boolean;
  status: string;
  primary_metric_label: string;
  primary_metric_value?: number | null;
  last_evaluated?: string | null;
}

export interface ModelComparisonMatrix {
  classification_comparison: Array<{
    model_id: string;
    name: string;
    evaluated: boolean;
    accuracy?: number | null;
    precision?: number | null;
    recall?: number | null;
    f1_score?: number | null;
    specificity?: number | null;
    sensitivity?: number | null;
    inference_time_ms?: number | null;
    parameters: string;
    status: string;
  }>;
  segmentation_comparison: Array<{
    model_id: string;
    name: string;
    evaluated: boolean;
    dice_coefficient?: number | null;
    mean_iou?: number | null;
    precision?: number | null;
    recall?: number | null;
    sensitivity?: number | null;
    specificity?: number | null;
    inference_time_ms?: number | null;
    parameters: string;
    status: string;
    evaluation_protocol: string;
  }>;
}

export interface EvaluationHistoryItem {
  run_id: string;
  model: string;
  category: string;
  dataset: string;
  evaluation_protocol?: string;
  accuracy?: number | null;
  macro_f1?: number | null;
  whole_tumor_dice?: number | null;
  whole_tumor_iou?: number | null;
  parameters?: string;
  date: string;
  verified: boolean;
  status: string;
}
