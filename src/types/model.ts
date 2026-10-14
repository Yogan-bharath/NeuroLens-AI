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
