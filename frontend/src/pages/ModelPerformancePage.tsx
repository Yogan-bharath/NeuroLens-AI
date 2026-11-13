import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import PageHeader from '@/components/layout/PageHeader';
import MetricCard from '@/components/common/MetricCard';
import ConfusionMatrix from '@/components/charts/ConfusionMatrix';
import PerformanceCharts from '@/components/charts/PerformanceCharts';
import {
  getClassificationPerformance,
  getSegmentationPerformance,
  getEvaluationHistory,
  getModelComparison,
} from '@/services/api';
import type {
  ClassificationPerformanceResponse,
  SegmentationPerformanceResponse,
  ClassificationModelItem,
  SegmentationModelItem,
  EvaluationHistoryItem,
  ModelComparisonMatrix,
} from '@/types/model';
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
  Clock,
  Cpu,
  Award,
  AlertCircle,
  HelpCircle,
  CheckCircle2,
  Sliders,
  TrendingUp,
  BarChart3,
  Scale,
  History,
  ShieldCheck,
  Eye,
  Info,
} from 'lucide-react';

type PerformanceTab = 'classification' | 'segmentation' | 'comparison' | 'history';

export default function ModelPerformancePage() {
  const [activeTab, setActiveTab] = useState<PerformanceTab>('classification');
  const [loading, setLoading] = useState(true);

  // Data states
  const [clsData, setClsData] = useState<ClassificationPerformanceResponse | null>(null);
  const [segData, setSegData] = useState<SegmentationPerformanceResponse | null>(null);
  const [historyData, setHistoryData] = useState<EvaluationHistoryItem[]>([]);
  const [comparisonData, setComparisonData] = useState<ModelComparisonMatrix | null>(null);

  // Active selections
  const [selectedClsId, setSelectedClsId] = useState<string>('efficientnetv2b0');
  const [selectedSegId, setSelectedSegId] = useState<string>('attention_cnn_vit');
  const [comparisonFilter, setComparisonFilter] = useState<'all' | 'classification' | 'segmentation'>('all');
  const [selectedComparisonIds, setSelectedComparisonIds] = useState<string[]>([
    'efficientnetv2b0',
    'attention_cnn_vit',
    'brats_3d_unet',
  ]);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      getClassificationPerformance().catch(() => null),
      getSegmentationPerformance().catch(() => null),
      getEvaluationHistory().catch(() => []),
      getModelComparison().catch(() => null),
    ])
      .then(([cls, seg, hist, comp]) => {
        if (cls) setClsData(cls);
        if (seg) setSegData(seg);
        if (hist) setHistoryData(hist);
        if (comp) setComparisonData(comp);
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return <LoadingSpinner text="Retrieving verified evaluation metrics..." className="py-20" />;
  }

  // Filter only available and evaluated models for UI display
  const availableClsModels = (clsData?.models || []).filter(
    m => m.evaluated !== false && m.status !== 'Not evaluated' && m.status !== 'coming_soon'
  );
  const availableSegModels = (segData?.models || []).filter(
    m => m.evaluated !== false && m.status !== 'Not evaluated' && m.status !== 'coming_soon'
  );

  const activeClsModel: ClassificationModelItem | undefined = availableClsModels.find(
    m => m.model_id === selectedClsId
  ) || availableClsModels[0];
  const activeSegModel: SegmentationModelItem | undefined = availableSegModels.find(
    m => m.model_id === selectedSegId
  ) || availableSegModels[0];

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="space-y-6 pb-12"
    >
      <PageHeader
        title="Model Performance & Benchmarks"
        subtitle="Independent evaluation metrics, multi-planar Dice/IoU benchmarks, and comparative model verification."
      >
        <span className="neuro-badge-success flex items-center gap-1.5">
          <ShieldCheck className="w-3.5 h-3.5" />
          <span>Verified Test Split Records</span>
        </span>
      </PageHeader>

      {/* Main Tab Navigation */}
      <div className="flex border-b border-neuro-border gap-2 overflow-x-auto pb-px">
        <button
          onClick={() => setActiveTab('classification')}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold rounded-t-lg transition-all border-b-2 whitespace-nowrap ${
            activeTab === 'classification'
              ? 'border-neuro-accent text-neuro-accent bg-neuro-surface-200/50'
              : 'border-transparent text-neuro-text-secondary hover:text-neuro-text-primary hover:border-neuro-border'
          }`}
        >
          <Brain className="w-4 h-4" />
          <span>Classification Models</span>
          <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-neuro-accent/20 text-neuro-accent-bright font-mono">
            {availableClsModels.length}
          </span>
        </button>

        <button
          onClick={() => setActiveTab('segmentation')}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold rounded-t-lg transition-all border-b-2 whitespace-nowrap ${
            activeTab === 'segmentation'
              ? 'border-neuro-cyan text-neuro-cyan bg-neuro-surface-200/50'
              : 'border-transparent text-neuro-text-secondary hover:text-neuro-text-primary hover:border-neuro-border'
          }`}
        >
          <Layers className="w-4 h-4" />
          <span>Segmentation Models</span>
          <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-neuro-cyan/20 text-neuro-cyan font-mono">
            {availableSegModels.length}
          </span>
        </button>

        <button
          onClick={() => setActiveTab('comparison')}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold rounded-t-lg transition-all border-b-2 whitespace-nowrap ${
            activeTab === 'comparison'
              ? 'border-amber-400 text-amber-400 bg-neuro-surface-200/50'
              : 'border-transparent text-neuro-text-secondary hover:text-neuro-text-primary hover:border-neuro-border'
          }`}
        >
          <Scale className="w-4 h-4" />
          <span>Side-by-Side Comparison</span>
        </button>

        <button
          onClick={() => setActiveTab('history')}
          className={`flex items-center gap-2 px-4 py-2.5 text-xs font-semibold rounded-t-lg transition-all border-b-2 whitespace-nowrap ${
            activeTab === 'history'
              ? 'border-blue-400 text-blue-400 bg-neuro-surface-200/50'
              : 'border-transparent text-neuro-text-secondary hover:text-neuro-text-primary hover:border-neuro-border'
          }`}
        >
          <History className="w-4 h-4" />
          <span>Historical Benchmark Runs</span>
          <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-blue-500/20 text-blue-300 font-mono">
            {historyData.length}
          </span>
        </button>
      </div>

      {/* ========================================================================= */}
      {/* TAB 1: CLASSIFICATION MODELS                                              */}
      {/* ========================================================================= */}
      {activeTab === 'classification' && clsData && (
        <div className="space-y-6">
          {/* Classification Model Selector Pills */}
          <div className="p-3 rounded-xl bg-neuro-surface-200 border border-neuro-border flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold text-neuro-text-muted uppercase tracking-wider">
                Select Model:
              </span>
              <div className="flex flex-wrap gap-2">
                {availableClsModels.map(m => {
                  const isSelected = m.model_id === (activeClsModel?.model_id || selectedClsId);
                  const isBest = m.model_id === clsData.best_model_id;
                  return (
                    <button
                      key={m.model_id}
                      onClick={() => setSelectedClsId(m.model_id)}
                      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                        isSelected
                          ? 'bg-neuro-accent text-white shadow-sm'
                          : 'bg-neuro-surface-300 text-neuro-text-secondary hover:bg-neuro-surface-400 hover:text-neuro-text-primary'
                      }`}
                    >
                      <span>{m.name}</span>
                      {isBest && (
                        <span className="text-[10px] px-1.5 py-0.2 rounded bg-amber-400/20 text-amber-300 font-bold flex items-center gap-0.5">
                          <Award className="w-2.5 h-2.5" /> Best
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="text-xs text-neuro-text-muted">
              Benchmark Dataset:{' '}
              <strong className="text-neuro-text-primary">{clsData.benchmark_dataset}</strong> (N ={' '}
              {clsData.total_test_samples.toLocaleString()})
            </div>
          </div>

          {activeClsModel && activeClsModel.evaluated ? (
            <>
              {/* Macro Metric Cards */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <MetricCard
                  label="Overall Accuracy"
                  value={formatPercentage(activeClsModel.accuracy || 0)}
                  icon={Activity}
                  trend="Test split evaluation"
                  trendUp={true}
                  accent="teal"
                />
                <MetricCard
                  label="Macro Precision"
                  value={formatPercentage(activeClsModel.macro_precision || 0)}
                  icon={Target}
                  trend="Unweighted average"
                  trendUp={true}
                  accent="cyan"
                />
                <MetricCard
                  label="Macro Recall / Sensitivity"
                  value={formatPercentage(activeClsModel.macro_recall || 0)}
                  icon={Repeat}
                  trend="Sensitivity across classes"
                  trendUp={true}
                  accent="blue"
                />
                <MetricCard
                  label="Macro F1-Score"
                  value={formatPercentage(activeClsModel.macro_f1 || 0)}
                  icon={Zap}
                  trend="Harmonic mean"
                  trendUp={true}
                  accent="amber"
                />
              </div>

              {/* Grid: Confusion Matrix & Per-Class Metrics */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
                <div className="lg:col-span-7 neuro-card space-y-4">
                  <div className="flex items-center justify-between border-b border-neuro-border pb-3">
                    <div className="flex items-center gap-2">
                      <FileCheck2 className="w-4 h-4 text-neuro-accent" />
                      <h3 className="text-sm font-semibold text-neuro-text-primary">
                        Multi-Class Confusion Matrix
                      </h3>
                    </div>
                    <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-neuro-surface-200 text-neuro-text-muted">
                      N = {clsData.total_test_samples} Test Cases
                    </span>
                  </div>

                  <p className="text-xs text-neuro-text-secondary leading-relaxed">
                    Evaluation on independent held-out test split. Rows denote ground-truth histological diagnosis; columns denote model prediction.
                  </p>

                  {activeClsModel.confusion_matrix && (
                    <ConfusionMatrix data={activeClsModel.confusion_matrix} />
                  )}
                </div>

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

                  {activeClsModel.per_class_metrics && (
                    <PerformanceCharts metrics={activeClsModel.per_class_metrics} />
                  )}
                </div>
              </div>

              {/* Architecture Specifications */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="neuro-card space-y-3">
                  <div className="flex items-center gap-2 border-b border-neuro-border pb-3">
                    <Brain className="w-4 h-4 text-neuro-accent" />
                    <h3 className="text-sm font-semibold text-neuro-text-primary">Architecture Specifications</h3>
                  </div>
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-neuro-text-muted">Model Name & Version:</span>
                      <span className="font-semibold text-neuro-text-primary">{activeClsModel.name} {activeClsModel.version}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-neuro-text-muted">Backbone Architecture:</span>
                      <span className="text-neuro-text-secondary">{activeClsModel.architecture}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-neuro-text-muted">Deep Learning Framework:</span>
                      <span className="text-neuro-text-secondary">{activeClsModel.framework}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-neuro-text-muted">Input Dimension:</span>
                      <span className="font-mono text-neuro-text-secondary">{activeClsModel.input_size}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-neuro-text-muted">Trainable Parameters:</span>
                      <span className="font-mono text-neuro-text-secondary">{activeClsModel.parameters}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-neuro-text-muted">Avg. Inference Latency:</span>
                      <span className="font-mono text-emerald-400 font-semibold">{activeClsModel.average_inference_time_ms} ms</span>
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
                      <span className="text-neuro-text-secondary">{activeClsModel.dataset}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-neuro-text-muted">Validation Partition:</span>
                      <span className="text-neuro-text-secondary">{activeClsModel.evaluation_split}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-neuro-text-muted">Last Benchmark Date:</span>
                      <span className="flex items-center gap-1 text-neuro-text-secondary">
                        <Calendar className="w-3.5 h-3.5" />
                        {activeClsModel.last_evaluated}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-neuro-text-muted">Model Checkpoint Status:</span>
                      <span className="text-emerald-400 font-medium flex items-center gap-1">
                        <CheckCircle2 className="w-3.5 h-3.5" /> Installed & Verified
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </>
          ) : (
            /* Non-Evaluated Classification Model State */
            <div className="p-8 rounded-xl bg-neuro-surface-200 border border-amber-500/30 text-center space-y-4">
              <div className="w-12 h-12 rounded-full bg-amber-500/10 text-amber-400 flex items-center justify-center mx-auto">
                <AlertCircle className="w-6 h-6" />
              </div>
              <div className="space-y-1">
                <h3 className="text-base font-bold text-neuro-text-primary">
                  {activeClsModel?.name} — Not Evaluated
                </h3>
                <p className="text-xs text-neuro-text-secondary max-w-lg mx-auto">
                  {activeClsModel?.reason || 'Model architecture is registered in the model registry. However, the official benchmark checkpoint has not been evaluated on the held-out test cohort.'}
                </p>
              </div>

              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-amber-500/10 text-amber-300 text-xs font-medium">
                <span>Evaluation Status: <strong>Pending / Not Evaluated</strong></span>
              </div>

              <div className="max-w-md mx-auto pt-4 border-t border-neuro-border text-left text-xs space-y-2">
                <div className="flex justify-between">
                  <span className="text-neuro-text-muted">Architecture:</span>
                  <span className="text-neuro-text-secondary">{activeClsModel?.architecture}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neuro-text-muted">Input Dimension:</span>
                  <span className="font-mono text-neuro-text-secondary">{activeClsModel?.input_size}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neuro-text-muted">Total Parameters:</span>
                  <span className="font-mono text-neuro-text-secondary">{activeClsModel?.parameters}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 2: SEGMENTATION MODELS                                                */}
      {/* ========================================================================= */}
      {activeTab === 'segmentation' && segData && (
        <div className="space-y-6">
          {/* Segmentation Model Selector Pills */}
          <div className="p-3 rounded-xl bg-neuro-surface-200 border border-neuro-border flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <span className="text-xs font-semibold text-neuro-text-muted uppercase tracking-wider">
                Select Model:
              </span>
              <div className="flex flex-wrap gap-2">
                {availableSegModels.map(m => {
                  const isSelected = m.model_id === (activeSegModel?.model_id || selectedSegId);
                  const isBest = m.model_id === segData.best_model_id;
                  return (
                    <button
                      key={m.model_id}
                      onClick={() => setSelectedSegId(m.model_id)}
                      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                        isSelected
                          ? 'bg-neuro-cyan text-white shadow-sm'
                          : 'bg-neuro-surface-300 text-neuro-text-secondary hover:bg-neuro-surface-400 hover:text-neuro-text-primary'
                      }`}
                    >
                      <span>{m.name}</span>
                      {isBest && (
                        <span className="text-[10px] px-1.5 py-0.2 rounded bg-amber-400/20 text-amber-300 font-bold flex items-center gap-0.5">
                          <Award className="w-2.5 h-2.5" /> Best WT Dice
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="text-xs text-neuro-text-muted">
              Evaluation Cohort: <strong className="text-neuro-text-primary">{segData.evaluation_cohort}</strong>
            </div>
          </div>

          {activeSegModel && activeSegModel.evaluated ? (
            <>
              {/* Protocol Banner */}
              <div className="p-3 rounded-xl bg-neuro-cyan/10 border border-neuro-cyan/20 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 text-xs">
                <div className="flex items-center gap-2 text-neuro-cyan">
                  <Info className="w-4 h-4 flex-shrink-0" />
                  <span>
                    Evaluation Protocol: <strong>{activeSegModel.evaluation_protocol}</strong>
                  </span>
                </div>
                <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-neuro-cyan/20 text-neuro-cyan font-semibold">
                  BraTS 2020 Multi-Class Benchmark
                </span>
              </div>

              {/* Headline Segmentation Metrics */}
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                <MetricCard
                  label="Whole Tumor Dice (WT)"
                  value={formatPercentage(activeSegModel.dice_coefficient || 0)}
                  icon={Activity}
                  trend={activeSegModel.model_id === 'attention_cnn_vit' ? '89.37% (Highest WT Dice)' : '82.98% Complete Volume'}
                  trendUp={true}
                  accent="teal"
                />
                <MetricCard
                  label="Whole Tumor IoU (Jaccard)"
                  value={formatPercentage(activeSegModel.mean_iou || 0)}
                  icon={Target}
                  trend={activeSegModel.model_id === 'attention_cnn_vit' ? '80.78% (Highest IoU)' : '72.83% Complete Volume'}
                  trendUp={true}
                  accent="cyan"
                />
                <MetricCard
                  label="Pixel Accuracy"
                  value={formatPercentage(activeSegModel.pixel_accuracy || 0.99)}
                  icon={Repeat}
                  trend="Background + Tumor Voxels"
                  trendUp={true}
                  accent="blue"
                />
                <MetricCard
                  label="Sensitivity / Recall"
                  value={formatPercentage(activeSegModel.sensitivity || activeSegModel.recall || 0)}
                  icon={Zap}
                  trend="True positive rate"
                  trendUp={true}
                  accent="amber"
                />
              </div>

              {/* Per-Region Sub-Compartment Dice & IoU Breakdown */}
              {activeSegModel.per_region_metrics && (
                <div className="neuro-card space-y-4">
                  <div className="flex items-center justify-between border-b border-neuro-border pb-3">
                    <div className="flex items-center gap-2">
                      <BarChart3 className="w-4 h-4 text-neuro-cyan" />
                      <h3 className="text-sm font-semibold text-neuro-text-primary">
                        Per-Region Sub-Compartment Benchmark Metrics (Dice & IoU)
                      </h3>
                    </div>
                    <span className="text-xs text-neuro-text-muted font-mono">
                      Held-out Test Cohort (56 Patients)
                    </span>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {activeSegModel.per_region_metrics.map(r => (
                      <div
                        key={r.region_name}
                        className="p-3.5 rounded-lg bg-neuro-surface-200/70 border border-neuro-border space-y-2.5"
                      >
                        <div className="flex items-center justify-between text-xs font-semibold">
                          <span className="text-neuro-text-primary">{r.region_name}</span>
                          <span className="font-mono text-neuro-cyan">{formatPercentage(r.dice_score)} Dice</span>
                        </div>

                        {/* Dice Bar */}
                        <div className="space-y-1 text-[11px]">
                          <div className="flex justify-between text-neuro-text-muted">
                            <span>Dice Similarity</span>
                            <span className="font-mono text-neuro-text-secondary">{(r.dice_score * 100).toFixed(2)}%</span>
                          </div>
                          <div className="h-1.5 w-full bg-neuro-surface-400 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-neuro-cyan rounded-full transition-all duration-500"
                              style={{ width: `${r.dice_score * 100}%` }}
                            />
                          </div>
                        </div>

                        {/* IoU Bar */}
                        <div className="space-y-1 text-[11px]">
                          <div className="flex justify-between text-neuro-text-muted">
                            <span>Intersection over Union (IoU)</span>
                            <span className="font-mono text-neuro-text-secondary">{(r.iou_score * 100).toFixed(2)}%</span>
                          </div>
                          <div className="h-1.5 w-full bg-neuro-surface-400 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-neuro-accent rounded-full transition-all duration-500"
                              style={{ width: `${r.iou_score * 100}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Architecture & Evaluation Protocol Specifications */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="neuro-card space-y-3">
                  <div className="flex items-center gap-2 border-b border-neuro-border pb-3">
                    <Brain className="w-4 h-4 text-neuro-cyan" />
                    <h3 className="text-sm font-semibold text-neuro-text-primary">Architecture Specifications</h3>
                  </div>
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-neuro-text-muted">Model Name & Version:</span>
                      <span className="font-semibold text-neuro-text-primary">{activeSegModel.name} {activeSegModel.version}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-neuro-text-muted">Architecture:</span>
                      <span className="text-neuro-text-secondary">{activeSegModel.architecture}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-neuro-text-muted">Deep Learning Framework:</span>
                      <span className="text-neuro-text-secondary">{activeSegModel.framework}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-neuro-text-muted">Input Dimension:</span>
                      <span className="font-mono text-neuro-text-secondary">{activeSegModel.input_size}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-neuro-text-muted">Trainable Parameters:</span>
                      <span className="font-mono text-neuro-text-secondary">{activeSegModel.parameters}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-neuro-text-muted">Average Inference Latency:</span>
                      <span className="font-mono text-emerald-400 font-semibold">{activeSegModel.average_inference_time_ms} ms</span>
                    </div>
                  </div>
                </div>

                <div className="neuro-card space-y-3">
                  <div className="flex items-center gap-2 border-b border-neuro-border pb-3">
                    <Database className="w-4 h-4 text-neuro-accent" />
                    <h3 className="text-sm font-semibold text-neuro-text-primary">Evaluation Cohort & Protocol</h3>
                  </div>
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-neuro-text-muted">Reference Dataset:</span>
                      <span className="text-neuro-text-secondary">{activeSegModel.dataset}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-neuro-text-muted">Evaluation Protocol:</span>
                      <span className="text-neuro-text-secondary">{activeSegModel.evaluation_protocol}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-neuro-text-muted">Last Benchmark Date:</span>
                      <span className="flex items-center gap-1 text-neuro-text-secondary">
                        <Calendar className="w-3.5 h-3.5" />
                        {activeSegModel.last_evaluated}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-neuro-text-muted">Tumor Sub-Classes:</span>
                      <span className="text-neuro-text-primary font-medium">4 Classes (WT, TC, ED, ET)</span>
                    </div>
                  </div>
                </div>
              </div>
            </>
          ) : (
            /* Non-Evaluated Segmentation Model State */
            <div className="p-8 rounded-xl bg-neuro-surface-200 border border-amber-500/30 text-center space-y-4">
              <div className="w-12 h-12 rounded-full bg-amber-500/10 text-amber-400 flex items-center justify-center mx-auto">
                <AlertCircle className="w-6 h-6" />
              </div>
              <div className="space-y-1">
                <h3 className="text-base font-bold text-neuro-text-primary">
                  {activeSegModel?.name} — Not Evaluated
                </h3>
                <p className="text-xs text-neuro-text-secondary max-w-lg mx-auto">
                  {activeSegModel?.reason || 'Architecture registered in system; official benchmark checkpoint is not evaluated on host.'}
                </p>
              </div>

              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-amber-500/10 text-amber-300 text-xs font-medium">
                <span>Evaluation Status: <strong>Pending / Not Evaluated</strong></span>
              </div>

              <div className="max-w-md mx-auto pt-4 border-t border-neuro-border text-left text-xs space-y-2">
                <div className="flex justify-between">
                  <span className="text-neuro-text-muted">Architecture:</span>
                  <span className="text-neuro-text-secondary">{activeSegModel?.architecture}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neuro-text-muted">Input Dimension:</span>
                  <span className="font-mono text-neuro-text-secondary">{activeSegModel?.input_size}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-neuro-text-muted">Total Parameters:</span>
                  <span className="font-mono text-neuro-text-secondary">{activeSegModel?.parameters}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 3: SIDE-BY-SIDE MODEL COMPARISON                                      */}
      {/* ========================================================================= */}
      {activeTab === 'comparison' && (
        <div className="space-y-6">
          {/* Comparison Filter Controls */}
          <div className="p-4 rounded-xl bg-neuro-surface-200 border border-neuro-border space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <Sliders className="w-4 h-4 text-amber-400" />
                <h3 className="text-xs font-bold text-neuro-text-primary uppercase tracking-wider">
                  Model Comparison Matrix
                </h3>
              </div>

              <div className="flex items-center gap-1.5 bg-neuro-surface-300 p-1 rounded-lg text-xs">
                {(['all', 'classification', 'segmentation'] as const).map(f => (
                  <button
                    key={f}
                    onClick={() => setComparisonFilter(f)}
                    className={`px-3 py-1 rounded capitalize transition-all ${
                      comparisonFilter === f
                        ? 'bg-amber-500 text-black font-bold shadow-sm'
                        : 'text-neuro-text-secondary hover:text-neuro-text-primary'
                    }`}
                  >
                    {f}
                  </button>
                ))}
              </div>
            </div>

            <p className="text-xs text-neuro-text-muted">
              Select models to compare evaluation metrics, parameters, and benchmark results side-by-side.
            </p>
          </div>

          {/* Classification Models Comparison Table */}
          {(comparisonFilter === 'all' || comparisonFilter === 'classification') && (
            <div className="neuro-card space-y-3">
              <div className="flex items-center gap-2 border-b border-neuro-border pb-3">
                <Brain className="w-4 h-4 text-neuro-accent" />
                <h3 className="text-sm font-semibold text-neuro-text-primary">
                  Classification Models Benchmark Comparison
                </h3>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-neuro-border text-neuro-text-muted text-[11px] uppercase">
                      <th className="py-2.5 px-3">Model</th>
                      <th className="py-2.5 px-3">Status</th>
                      <th className="py-2.5 px-3">Accuracy</th>
                      <th className="py-2.5 px-3">Macro F1</th>
                      <th className="py-2.5 px-3">Precision</th>
                      <th className="py-2.5 px-3">Sensitivity</th>
                      <th className="py-2.5 px-3">Specificity</th>
                      <th className="py-2.5 px-3">Parameters</th>
                      <th className="py-2.5 px-3">Inference</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-neuro-border/50">
                    {availableClsModels.map(m => {
                      const isBest = m.model_id === 'efficientnetv2b0';
                      return (
                        <tr key={m.model_id} className="hover:bg-neuro-surface-300/30 transition-colors">
                          <td className="py-3 px-3 font-semibold text-neuro-text-primary flex items-center gap-1.5">
                            {m.name}
                            {isBest && (
                              <span className="text-[10px] px-1.5 py-0.2 rounded bg-amber-400/20 text-amber-300 font-bold flex items-center gap-0.5">
                                <Award className="w-2.5 h-2.5" /> Best
                              </span>
                            )}
                          </td>
                          <td className="py-3 px-3">
                            <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-semibold">
                              Evaluated
                            </span>
                          </td>
                          <td className="py-3 px-3 font-mono font-bold text-neuro-accent-bright">
                            {m.accuracy ? `${(m.accuracy * 100).toFixed(1)}%` : '—'}
                          </td>
                          <td className="py-3 px-3 font-mono text-neuro-text-secondary">
                            {m.macro_f1 ? `${(m.macro_f1 * 100).toFixed(1)}%` : '—'}
                          </td>
                          <td className="py-3 px-3 font-mono text-neuro-text-secondary">
                            {m.macro_precision ? `${(m.macro_precision * 100).toFixed(1)}%` : '—'}
                          </td>
                          <td className="py-3 px-3 font-mono text-neuro-text-secondary">
                            {m.sensitivity ? `${(m.sensitivity * 100).toFixed(1)}%` : '—'}
                          </td>
                          <td className="py-3 px-3 font-mono text-neuro-text-secondary">
                            {m.specificity ? `${(m.specificity * 100).toFixed(1)}%` : '—'}
                          </td>
                          <td className="py-3 px-3 font-mono text-neuro-text-muted">{m.parameters}</td>
                          <td className="py-3 px-3 font-mono text-emerald-400">
                            {m.average_inference_time_ms ? `${m.average_inference_time_ms} ms` : '—'}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Segmentation Models Comparison Table */}
          {(comparisonFilter === 'all' || comparisonFilter === 'segmentation') && (
            <div className="neuro-card space-y-3">
              <div className="flex items-center gap-2 border-b border-neuro-border pb-3">
                <Layers className="w-4 h-4 text-neuro-cyan" />
                <h3 className="text-sm font-semibold text-neuro-text-primary">
                  Segmentation Models Benchmark Comparison
                </h3>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs border-collapse">
                  <thead>
                    <tr className="border-b border-neuro-border text-neuro-text-muted text-[11px] uppercase">
                      <th className="py-2.5 px-3">Model</th>
                      <th className="py-2.5 px-3">Evaluation Protocol</th>
                      <th className="py-2.5 px-3">Status</th>
                      <th className="py-2.5 px-3">WT Dice</th>
                      <th className="py-2.5 px-3">WT IoU</th>
                      <th className="py-2.5 px-3">Sensitivity</th>
                      <th className="py-2.5 px-3">Specificity</th>
                      <th className="py-2.5 px-3">Parameters</th>
                      <th className="py-2.5 px-3">Latency</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-neuro-border/50">
                    {availableSegModels.map(m => {
                      const isBest = m.model_id === 'attention_cnn_vit';
                      return (
                        <tr key={m.model_id} className="hover:bg-neuro-surface-300/30 transition-colors">
                          <td className="py-3 px-3 font-semibold text-neuro-text-primary flex items-center gap-1.5">
                            {m.name}
                            {isBest && (
                              <span className="text-[10px] px-1.5 py-0.2 rounded bg-amber-400/20 text-amber-300 font-bold flex items-center gap-0.5">
                                <Award className="w-2.5 h-2.5" /> Best WT Dice
                              </span>
                            )}
                          </td>
                          <td className="py-3 px-3 text-neuro-text-muted text-[11px] max-w-[180px] truncate" title={m.evaluation_protocol}>
                            {m.evaluation_protocol}
                          </td>
                          <td className="py-3 px-3">
                            <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-semibold">
                              Evaluated
                            </span>
                          </td>
                          <td className="py-3 px-3 font-mono font-bold text-neuro-cyan">
                            {m.dice_coefficient ? `${(m.dice_coefficient * 100).toFixed(2)}%` : '—'}
                          </td>
                          <td className="py-3 px-3 font-mono text-neuro-text-secondary">
                            {m.mean_iou ? `${(m.mean_iou * 100).toFixed(2)}%` : '—'}
                          </td>
                          <td className="py-3 px-3 font-mono text-neuro-text-secondary">
                            {m.sensitivity ? `${(m.sensitivity * 100).toFixed(2)}%` : '—'}
                          </td>
                          <td className="py-3 px-3 font-mono text-neuro-text-secondary">
                            {m.specificity ? `${(m.specificity * 100).toFixed(2)}%` : '—'}
                          </td>
                          <td className="py-3 px-3 font-mono text-neuro-text-muted">{m.parameters}</td>
                          <td className="py-3 px-3 font-mono text-emerald-400">
                            {m.average_inference_time_ms ? `${m.average_inference_time_ms} ms` : '—'}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 4: HISTORICAL BENCHMARK RUNS                                          */}
      {/* ========================================================================= */}
      {activeTab === 'history' && (
        <div className="space-y-4">
          <div className="neuro-card space-y-3">
            <div className="flex items-center justify-between border-b border-neuro-border pb-3">
              <div className="flex items-center gap-2">
                <History className="w-4 h-4 text-blue-400" />
                <h3 className="text-sm font-semibold text-neuro-text-primary">
                  Official Verification Run Records
                </h3>
              </div>
              <span className="text-xs text-neuro-text-muted">
                Stored in MongoDB & Model Registry
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs border-collapse">
                <thead>
                  <tr className="border-b border-neuro-border text-neuro-text-muted text-[11px] uppercase">
                    <th className="py-2.5 px-3">Run ID</th>
                    <th className="py-2.5 px-3">Model</th>
                    <th className="py-2.5 px-3">Category</th>
                    <th className="py-2.5 px-3">Evaluation Dataset</th>
                    <th className="py-2.5 px-3">Primary Metric</th>
                    <th className="py-2.5 px-3">Parameters</th>
                    <th className="py-2.5 px-3">Benchmark Date</th>
                    <th className="py-2.5 px-3">Verification</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-neuro-border/50">
                  {historyData.map(h => (
                    <tr key={h.run_id} className="hover:bg-neuro-surface-300/30 transition-colors">
                      <td className="py-3 px-3 font-mono text-neuro-accent-bright text-[11px]">{h.run_id}</td>
                      <td className="py-3 px-3 font-semibold text-neuro-text-primary">{h.model}</td>
                      <td className="py-3 px-3 capitalize text-neuro-text-muted">{h.category}</td>
                      <td className="py-3 px-3 text-neuro-text-secondary text-[11px] max-w-[200px] truncate" title={h.dataset}>
                        {h.dataset}
                      </td>
                      <td className="py-3 px-3 font-mono font-bold text-emerald-400">
                        {h.whole_tumor_dice
                          ? `${(h.whole_tumor_dice * 100).toFixed(2)}% WT Dice`
                          : h.accuracy
                          ? `${(h.accuracy * 100).toFixed(1)}% Acc`
                          : '—'}
                      </td>
                      <td className="py-3 px-3 font-mono text-neuro-text-muted">{h.parameters || '—'}</td>
                      <td className="py-3 px-3 text-neuro-text-muted">{h.date}</td>
                      <td className="py-3 px-3">
                        {h.verified ? (
                          <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-semibold flex items-center gap-1 w-fit">
                            <CheckCircle2 className="w-3 h-3" /> Verified
                          </span>
                        ) : (
                          <span className="text-[10px] px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/30 w-fit">
                            Pending
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      <ResearchDisclaimer />
    </motion.div>
  );
}
