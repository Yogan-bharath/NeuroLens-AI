import type { ClassificationScore, AnalysisMetadata } from '@/types/analysis';
import ClassScoreChart from './ClassScoreChart';
import ConfidenceIndicator from './ConfidenceIndicator';
import { Brain, AlertCircle } from 'lucide-react';
import { ResearchDisclaimer } from '@/components/common/ResearchDisclaimer';

interface ClassificationResultsProps {
  predictedClass: string;
  confidence: number;
  scores: ClassificationScore[];
  metadata?: AnalysisMetadata | null;
}

export default function ClassificationResults({
  predictedClass,
  confidence,
  scores,
  metadata,
}: ClassificationResultsProps) {
  // Clinical research interpretation text based on predicted class
  const getInterpretation = (label: string) => {
    switch (label.toLowerCase()) {
      case 'glioma':
        return 'The neural network model detected feature patterns consistent with intra-axial glial neoplasms. These patterns commonly feature irregular border definitions and variable signal intensity.';
      case 'meningioma':
        return 'The model observed focal extra-axial characteristics typical of meningioma representations, including broad dural attachment morphology.';
      case 'pituitary':
        return 'Sellar region feature maps suggest expansion or signal variations characteristic of pituitary adenoma patterns in the training cohort.';
      case 'no tumor':
        return 'No distinct focal mass or abnormal hyperintensity patterns exceeding the classification threshold were flagged by the network.';
      default:
        return 'Classification inference completed based on learned deep convolutional representations.';
    }
  };

  return (
    <div className="neuro-card space-y-5">
      {/* Header and top prediction */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-neuro-border">
        <div>
          <span className="text-xs uppercase font-medium tracking-wider text-neuro-text-muted">
            Predicted Class
          </span>
          <h2 className="text-xl font-bold text-neuro-text-primary mt-0.5">
            {predictedClass}
          </h2>
          <div className="flex items-center gap-2 mt-1">
            <span className="neuro-badge-accent">
              NeuroClassifier {metadata?.modelVersion || 'v2.1.0'}
            </span>
            <span className="text-xs text-neuro-text-muted">
              {metadata?.processingTimeMs ? `${(metadata.processingTimeMs / 1000).toFixed(2)}s inference` : 'Simulated'}
            </span>
          </div>
        </div>

        <ConfidenceIndicator confidence={confidence} size="md" />
      </div>

      {/* Class distribution */}
      <div>
        <h4 className="text-xs font-semibold text-neuro-text-secondary uppercase tracking-wider mb-3">
          Class Probability Distribution
        </h4>
        <ClassScoreChart scores={scores} />
      </div>

      {/* Interpretation */}
      <div className="p-3 rounded-lg bg-neuro-surface-200/80 border border-neuro-border space-y-1.5">
        <div className="flex items-center gap-1.5 text-xs font-semibold text-neuro-text-primary">
          <Brain className="w-3.5 h-3.5 text-neuro-accent" />
          <span>Model Output Interpretation</span>
        </div>
        <p className="text-xs text-neuro-text-secondary leading-relaxed">
          {getInterpretation(predictedClass)}
        </p>
      </div>

      {/* Notice */}
      <div className="flex items-start gap-2 text-[11px] text-neuro-text-muted">
        <AlertCircle className="w-3.5 h-3.5 text-neuro-text-muted flex-shrink-0 mt-0.5" />
        <p>
          Class probabilities are statistical model outputs derived from ResNet feature maps, not calibrated clinical diagnostic probabilities.
        </p>
      </div>

      <ResearchDisclaimer />
    </div>
  );
}
