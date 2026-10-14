import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import PageHeader from '@/components/layout/PageHeader';
import MetricCard from '@/components/common/MetricCard';
import ConfusionMatrix from '@/components/charts/ConfusionMatrix';
import PerformanceCharts from '@/components/charts/PerformanceCharts';
import { mockGetModelMetrics } from '@/services/mockApi';
import type { ModelMetrics } from '@/types/model';
import { LoadingSpinner } from '@/components/common/LoadingState';
import { ResearchDisclaimer } from '@/components/common/ResearchDisclaimer';
import { formatPercentage } from '@/lib/utils';
import {
  Activity,
  Target,
  Repeat,
  Zap,
  Brain,
  Database,
  Calendar,
  Layers,
  FileCheck2,
} from 'lucide-react';

export default function ModelPerformancePage() {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    mockGetModelMetrics()
      .then(data => setMetrics(data))
      .finally(() => setLoading(false));
  }, []);

  if (loading || !metrics) {
    return <LoadingSpinner text="Retrieving evaluation metrics..." className="py-20" />;
  }

  const { metadata } = metrics;

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="space-y-6"
    >
      <PageHeader
        title="Model Performance"
        subtitle="Benchmark metrics, confusion matrix, and validation split results for the NeuroClassifier pipeline."
      >
        <span className="neuro-badge-warning">
          Example Evaluation Cohort
        </span>
      </PageHeader>

      {/* Top 4 Overall Macro Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Overall Accuracy"
          value={formatPercentage(metrics.accuracy)}
          icon={Activity}
          trend="Test split evaluation"
          trendUp={true}
          accent="teal"
        />
        <MetricCard
          label="Macro Precision"
          value={formatPercentage(metrics.macroPrecision)}
          icon={Target}
          trend="Unweighted average"
          trendUp={true}
          accent="cyan"
        />
        <MetricCard
          label="Macro Recall"
          value={formatPercentage(metrics.macroRecall)}
          icon={Repeat}
          trend="Sensitivity across classes"
          trendUp={true}
          accent="blue"
        />
        <MetricCard
          label="Macro F1-Score"
          value={formatPercentage(metrics.macroF1)}
          icon={Zap}
          trend="Harmonic mean"
          trendUp={true}
          accent="amber"
        />
      </div>

      {/* Grid: Confusion Matrix & Per-Class Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left: Confusion Matrix (7 cols) */}
        <div className="lg:col-span-7 neuro-card space-y-4">
          <div className="flex items-center justify-between border-b border-neuro-border pb-3">
            <div className="flex items-center gap-2">
              <FileCheck2 className="w-4 h-4 text-neuro-accent" />
              <h3 className="text-sm font-semibold text-neuro-text-primary">
                Multi-Class Confusion Matrix
              </h3>
            </div>
            <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-neuro-surface-200 text-neuro-text-muted">
              N = 1,311 Test Cases
            </span>
          </div>

          <p className="text-xs text-neuro-text-secondary leading-relaxed">
            Evaluation on independent held-out test split. Rows indicate ground-truth histological pathology; columns denote model predictions.
          </p>

          <ConfusionMatrix data={metrics.confusionMatrix} />
        </div>

        {/* Right: Per-Class Breakdown (5 cols) */}
        <div className="lg:col-span-5 neuro-card space-y-4">
          <div className="flex items-center justify-between border-b border-neuro-border pb-3">
            <div className="flex items-center gap-2">
              <Layers className="w-4 h-4 text-neuro-cyan" />
              <h3 className="text-sm font-semibold text-neuro-text-primary">
                Per-Class Performance
              </h3>
            </div>
            <span className="text-xs text-neuro-text-muted">Test Partition</span>
          </div>

          <PerformanceCharts metrics={metrics.perClassMetrics} />
        </div>
      </div>

      {/* Model & Dataset Metadata */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="neuro-card space-y-3">
          <div className="flex items-center gap-2 border-b border-neuro-border pb-3">
            <Brain className="w-4 h-4 text-neuro-accent" />
            <h3 className="text-sm font-semibold text-neuro-text-primary">Architecture Specifications</h3>
          </div>
          <div className="space-y-2 text-xs">
            <div className="flex justify-between">
              <span className="text-neuro-text-muted">Model Name & Version:</span>
              <span className="font-semibold text-neuro-text-primary">{metadata.name} {metadata.version}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-neuro-text-muted">Backbone Architecture:</span>
              <span className="text-neuro-text-secondary">{metadata.architecture}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-neuro-text-muted">Deep Learning Framework:</span>
              <span className="text-neuro-text-secondary">{metadata.framework}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-neuro-text-muted">Input Dimension:</span>
              <span className="font-mono text-neuro-text-secondary">{metadata.inputSize}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-neuro-text-muted">Trainable Parameters:</span>
              <span className="font-mono text-neuro-text-secondary">{metadata.totalParameters}</span>
            </div>
          </div>
        </div>

        <div className="neuro-card space-y-3">
          <div className="flex items-center gap-2 border-b border-neuro-border pb-3">
            <Database className="w-4 h-4 text-neuro-cyan" />
            <h3 className="text-sm font-semibold text-neuro-text-primary">Dataset & Evaluation Protocol</h3>
          </div>
          <div className="space-y-2 text-xs">
            <div className="flex justify-between">
              <span className="text-neuro-text-muted">Reference Dataset:</span>
              <span className="text-neuro-text-secondary">{metadata.trainingDataset}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-neuro-text-muted">Validation Partition:</span>
              <span className="text-neuro-text-secondary">{metadata.evaluationSplit}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-neuro-text-muted">Benchmark Date:</span>
              <span className="flex items-center gap-1 text-neuro-text-secondary">
                <Calendar className="w-3.5 h-3.5" />
                {metadata.lastUpdated}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-neuro-text-muted">Data Integrity:</span>
              <span className="text-amber-400 font-medium">Demonstration Benchmark</span>
            </div>
          </div>
        </div>
      </div>

      <ResearchDisclaimer />
    </motion.div>
  );
}
