import { useRef } from 'react';
import { motion } from 'framer-motion';
import PageHeader from '@/components/layout/PageHeader';
import AnalysisModeSelector from '@/components/config/AnalysisModeSelector';
import ClassificationModelSelector from '@/components/config/ClassificationModelSelector';
import SegmentationModelSelector from '@/components/config/SegmentationModelSelector';
import XaiMethodSelector from '@/components/config/XaiMethodSelector';
import AdvancedExplainabilitySettings from '@/components/config/AdvancedExplainabilitySettings';
import SegmentationXaiSelector from '@/components/config/SegmentationXaiSelector';
import ConfigurationSummary from '@/components/config/ConfigurationSummary';
import CompatibilityWarning from '@/components/config/CompatibilityWarning';
import { ResearchDisclaimer } from '@/components/common/ResearchDisclaimer';
import { useConfiguration } from '@/context/ConfigurationContext';
import {
  classificationModelsRegistry,
  segmentationModelsRegistry,
} from '@/services/modelRegistry';
import { Sliders, Cpu, Sparkles, FileCheck2 } from 'lucide-react';

const workflowSteps = [
  { step: '1', title: 'Analysis Type', icon: Cpu },
  { step: '2', title: 'Model Selection', icon: Sliders },
  { step: '3', title: 'XAI Configuration', icon: Sparkles },
  { step: '4', title: 'Review & Analyze', icon: FileCheck2 },
];

export default function ModelSelectionPage() {
  const {
    config,
    setMode,
    setClassificationModel,
    setSegmentationModel,
    setClassificationXaiMethod,
    setClassificationLayer,
    setSegmentationXaiMethod,
    setSegmentationLayer,
    currentClassificationModel,
    currentSegmentationModel,
    currentXaiMethod,
    currentSegmentationXaiMethod,
    availableLayers,
    availableSegmentationLayers,
    supportedSegmentationXaiMethods,
    validation,
    compatibilityMessage,
    clearCompatibilityMessage,
  } = useConfiguration();

  const reviewRef = useRef<HTMLDivElement>(null);

  const showClassification = config.mode === 'classification' || config.mode === 'combined';
  const showSegmentation = config.mode === 'segmentation' || config.mode === 'combined';

  const scrollToReview = () => {
    reviewRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="space-y-8 max-w-5xl mx-auto"
    >
      <PageHeader
        title="Choose Your Analysis Configuration"
        subtitle="Select the models and explanation methods for your brain MRI analysis."
      />

      {/* Progress Workflow Indicator */}
      <div className="neuro-panel p-3 bg-neuro-surface-200/60 border border-neuro-border">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {workflowSteps.map(item => (
            <div
              key={item.step}
              className="flex items-center gap-2.5 p-2 rounded-md bg-neuro-surface/50 border border-neuro-border/50 text-xs"
            >
              <div className="w-6 h-6 rounded-full bg-neuro-accent text-neuro-bg flex items-center justify-center font-bold text-xs flex-shrink-0">
                {item.step}
              </div>
              <div className="min-w-0">
                <span className="font-semibold text-neuro-text-primary block truncate">
                  {item.title}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Compatibility Notice banner */}
      {compatibilityMessage && (
        <CompatibilityWarning
          message={compatibilityMessage}
          onDismiss={clearCompatibilityMessage}
        />
      )}

      {/* ── Step 1: Analysis Mode ── */}
      <section className="space-y-4">
        <AnalysisModeSelector
          selectedMode={config.mode}
          onSelectMode={setMode}
        />
      </section>

      {/* ── Step 2: Model Selection ── */}
      <section className="space-y-6">
        {showClassification && (
          <div className="space-y-4">
            <ClassificationModelSelector
              models={classificationModelsRegistry}
              selectedModelId={config.classificationModelId}
              onSelectModel={setClassificationModel}
            />
          </div>
        )}

        {showSegmentation && (
          <div className="space-y-4">
            <SegmentationModelSelector
              models={segmentationModelsRegistry}
              selectedModelId={config.segmentationModelId}
              onSelectModel={setSegmentationModel}
            />
          </div>
        )}
      </section>

      {/* ── Step 3: XAI Configuration ── */}
      <section className="space-y-6">
        {showClassification && (
          <div className="space-y-4">
            <XaiMethodSelector
              selectedMethodId={config.classificationXaiMethod}
              onSelectMethod={setClassificationXaiMethod}
              selectedModel={currentClassificationModel}
            />

            {/* Advanced layer selection for Classification */}
            <AdvancedExplainabilitySettings
              layers={availableLayers}
              selectedLayerId={config.classificationLayer}
              onSelectLayer={setClassificationLayer}
              xaiMethod={currentXaiMethod}
              modelName={currentClassificationModel?.name || 'Selected Model'}
            />
          </div>
        )}

        {showSegmentation && (
          <div className="space-y-4">
            {/* Step 3 (B): Segmentation XAI Method & Layer */}
            <SegmentationXaiSelector
              methods={supportedSegmentationXaiMethods}
              selectedMethodId={config.segmentationXaiMethod}
              onSelectMethod={setSegmentationXaiMethod}
              layers={availableSegmentationLayers}
              selectedLayerId={config.segmentationLayer}
              onSelectLayer={setSegmentationLayer}
              segmentationModel={currentSegmentationModel}
            />
          </div>
        )}
      </section>

      {/* ── Step 4: Review Configuration ── */}
      <section ref={reviewRef} className="space-y-4 pt-2">
        <ConfigurationSummary
          config={config}
          classificationModel={currentClassificationModel}
          segmentationModel={currentSegmentationModel}
          xaiMethod={currentXaiMethod}
          segmentationXaiMethod={currentSegmentationXaiMethod}
          validation={validation}
          onBackToConfig={scrollToReview}
        />
      </section>

      {/* Medical AI Research Disclaimer */}
      <ResearchDisclaimer />
    </motion.div>
  );
}
