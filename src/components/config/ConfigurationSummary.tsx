import { useNavigate } from 'react-router-dom';
import type {
  AnalysisConfiguration,
  ClassificationModel,
  SegmentationModel,
  XaiMethod,
} from '@/types/modelRegistry';
import type { ValidationResult } from '@/services/modelRegistry';
import { formatPercentage } from '@/lib/utils';
import {
  CheckCircle2,
  AlertCircle,
  ArrowRight,
  Brain,
  Layers,
  Sparkles,
  FileCheck2,
} from 'lucide-react';

interface ConfigurationSummaryProps {
  config: AnalysisConfiguration;
  classificationModel: ClassificationModel | undefined;
  segmentationModel: SegmentationModel | undefined;
  xaiMethod: XaiMethod | undefined;
  segmentationXaiMethod?: XaiMethod | undefined;
  validation: ValidationResult;
  onBackToConfig?: () => void;
}

export default function ConfigurationSummary({
  config,
  classificationModel,
  segmentationModel,
  xaiMethod,
  segmentationXaiMethod,
  validation,
  onBackToConfig,
}: ConfigurationSummaryProps) {
  const navigate = useNavigate();

  const handleProceed = () => {
    if (!validation.isValid) return;
    // Navigate to new-analysis page to proceed with upload using this configuration
    navigate('/new-analysis');
  };

  const showClassification = config.mode === 'classification' || config.mode === 'combined';
  const showSegmentation = config.mode === 'segmentation' || config.mode === 'combined';

  return (
    <div className="neuro-card p-5 space-y-5 border border-neuro-border bg-neuro-surface shadow-neuro-sm">
      <div className="flex items-center justify-between border-b border-neuro-border pb-3">
        <div className="flex items-center gap-2">
          <FileCheck2 className="w-5 h-5 text-neuro-accent" />
          <h3 className="text-sm font-bold text-neuro-text-primary uppercase tracking-wider">
            Step 4: Review Analysis Configuration
          </h3>
        </div>

        <span className="neuro-badge-accent uppercase text-[10px]">
          Mode: {config.mode}
        </span>
      </div>

      {/* Grid of configured pipelines */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
        {/* Classification Summary Card */}
        {showClassification ? (
          <div className="p-3.5 rounded-lg bg-neuro-surface-200/70 border border-neuro-border space-y-2.5">
            <div className="flex items-center gap-2 text-neuro-accent-bright font-semibold">
              <Brain className="w-4 h-4" />
              <span>Classification Configuration</span>
            </div>

            <div className="space-y-1.5 text-neuro-text-secondary">
              <div className="flex justify-between">
                <span className="text-neuro-text-muted">Model:</span>
                <span className="font-semibold text-neuro-text-primary">
                  {classificationModel?.name || config.classificationModelId} ({classificationModel?.version})
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-neuro-text-muted">Architecture:</span>
                <span className="truncate max-w-[180px]">{classificationModel?.architecture}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-neuro-text-muted">Test Accuracy:</span>
                <span className="font-mono text-neuro-text-primary">
                  {classificationModel?.metrics?.testAccuracy
                    ? formatPercentage(classificationModel.metrics.testAccuracy)
                    : 'Not evaluated'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-neuro-text-muted">XAI Method:</span>
                <span className="font-semibold text-neuro-cyan">
                  {xaiMethod?.name || config.classificationXaiMethod}
                </span>
              </div>
              {xaiMethod?.supportsLayerSelection && (
                <div className="flex justify-between">
                  <span className="text-neuro-text-muted">Feature Layer:</span>
                  <span className="font-mono text-neuro-accent-bright">
                    {config.classificationLayer}
                  </span>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="p-3.5 rounded-lg bg-neuro-surface-200/40 border border-neuro-border/40 text-neuro-text-muted italic flex items-center justify-center">
            Classification pipeline bypassed for this analysis mode.
          </div>
        )}

        {/* Segmentation Summary Card */}
        {showSegmentation ? (
          <div className="p-3.5 rounded-lg bg-neuro-surface-200/70 border border-neuro-border space-y-2.5">
            <div className="flex items-center gap-2 text-neuro-cyan font-semibold">
              <Layers className="w-4 h-4" />
              <span>Segmentation Configuration</span>
            </div>

            <div className="space-y-1.5 text-neuro-text-secondary">
              <div className="flex justify-between">
                <span className="text-neuro-text-muted">Model:</span>
                <span className="font-semibold text-neuro-text-primary">
                  {segmentationModel?.name || config.segmentationModelId}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-neuro-text-muted">Dice Coefficient:</span>
                <span className="font-mono text-neuro-text-primary">
                  {segmentationModel?.metrics?.diceCoefficient
                    ? formatPercentage(segmentationModel.metrics.diceCoefficient)
                    : 'Not evaluated'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-neuro-text-muted">Segmentation XAI:</span>
                <span className="font-semibold text-neuro-cyan">
                  {segmentationXaiMethod?.name || 'Mask-Targeted Grad-CAM'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-neuro-text-muted">XAI Target Layer:</span>
                <span className="font-mono text-neuro-cyan">
                  {config.segmentationLayer}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-neuro-text-muted">Viewer Layout:</span>
                <span className="text-neuro-text-primary font-medium">
                  Side-by-Side (Default)
                </span>
              </div>
            </div>
          </div>
        ) : (
          <div className="p-3.5 rounded-lg bg-neuro-surface-200/40 border border-neuro-border/40 text-neuro-text-muted italic flex items-center justify-center">
            Segmentation pipeline bypassed for this analysis mode.
          </div>
        )}
      </div>

      {/* Validation feedback */}
      {!validation.isValid && (
        <div className="p-3 rounded-lg bg-neuro-error/10 border border-neuro-error/20 text-xs space-y-1 text-red-300">
          <div className="flex items-center gap-1.5 font-semibold text-red-400">
            <AlertCircle className="w-4 h-4" />
            <span>Configuration Incomplete or Invalid</span>
          </div>
          <ul className="list-disc pl-5 space-y-0.5 text-[11px]">
            {validation.errors.map((err, i) => (
              <li key={i}>{err}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-2 border-t border-neuro-border">
        {onBackToConfig ? (
          <button
            type="button"
            onClick={onBackToConfig}
            className="neuro-btn-secondary text-xs w-full sm:w-auto"
          >
            Adjust Options
          </button>
        ) : (
          <div className="text-[11px] text-neuro-text-muted">
            Ready to apply configuration to analysis session.
          </div>
        )}

        <button
          type="button"
          disabled={!validation.isValid}
          onClick={handleProceed}
          className="neuro-btn-primary text-xs w-full sm:w-auto py-2.5 px-5 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          <span>Continue to MRI Upload / Analyze</span>
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
