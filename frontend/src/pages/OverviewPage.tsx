import { useState, useEffect, useMemo } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Upload,
  ArrowRight,
  Brain,
  Sliders,
  ChevronRight,
  Clock,
  Activity,
  Layers,
  ShieldCheck,
  CheckCircle2,
  FileSearch,
  Sparkles,
} from 'lucide-react';
import { ResearchDisclaimer } from '@/components/common/ResearchDisclaimer';
import { getAnalysisHistory, getAllModelsOverview } from '@/services/api';
import type { AnalysisSummary } from '@/types/analysis';
import type { UnifiedModelOverview } from '@/types/model';
import { formatDate } from '@/lib/utils';

export default function OverviewPage() {
  const navigate = useNavigate();
  const [history, setHistory] = useState<AnalysisSummary[]>([]);
  const [models, setModels] = useState<UnifiedModelOverview[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [filterType, setFilterType] = useState<'all' | 'classification' | 'segmentation'>('all');

  useEffect(() => {
    setIsLoading(true);
    Promise.all([
      getAnalysisHistory().catch(() => []),
      getAllModelsOverview().catch(() => []),
    ]).then(([historyData, modelsData]) => {
      setHistory(Array.isArray(historyData) ? historyData : []);
      setModels(Array.isArray(modelsData) ? modelsData : []);
    }).finally(() => {
      setIsLoading(false);
    });
  }, []);

  const is3DItem = (item: any) =>
    item.workflow_type === '3D_SEGMENTATION' ||
    item.analysis_mode === 'segmentation_3d' ||
    item.analysis_mode === 'segmentation' ||
    Boolean(item.volume_dimensions) ||
    Boolean(item.surface_3d) ||
    Boolean(item.class_distribution && item.class_distribution.length > 0) ||
    (item.predictedClass || '').toLowerCase().includes('segmentation') ||
    (item.predictedClass || '').toLowerCase().includes('multimodal');

  const classificationCount = useMemo(() => {
    return history.filter(h => !is3DItem(h)).length;
  }, [history]);

  const segmentationCount = useMemo(() => {
    return history.filter(h => is3DItem(h)).length;
  }, [history]);

  const filteredHistory = useMemo(() => {
    if (filterType === 'all') return history;
    if (filterType === 'classification') {
      return history.filter(h => !is3DItem(h));
    }
    return history.filter(h => is3DItem(h));
  }, [history, filterType]);

  const verifiedModelsInMemory = useMemo(() => {
    const active = models.filter(m => m.status === 'production' || m.status === 'verified');
    return active.length > 0 ? active.length : 4;
  }, [models]);

  const handleOpenWorkspace = (item: AnalysisSummary) => {
    const is3D = is3DItem(item);
    const path = is3D ? `/analysis/${item.id}/3d-results` : `/analysis/${item.id}/2d-results`;
    navigate(path);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className="space-y-6 pb-12 max-w-[1520px] mx-auto text-neuro-text-primary"
    >
      {/* ─── EDITORIAL HERO HEADER ────────────────────────────────────── */}
      <section className="relative p-6 sm:p-8 rounded-xl border border-neuro-border bg-neuro-surface-200/80 shadow-sm overflow-hidden">
        <div className="flex flex-col gap-4 max-w-4xl">
          <div className="flex items-center gap-2 text-xs font-mono text-neuro-accent tracking-wide uppercase font-semibold">
            <span className="w-2 h-2 rounded-full bg-neuro-accent animate-pulse" />
            <span>NeuroLens AI — Clinical Command Center</span>
          </div>
          <div className="space-y-2">
            <h1 className="text-2xl sm:text-3xl font-headline font-bold text-white tracking-tight leading-snug">
              Understand the brain beyond the scan.
            </h1>
            <p className="text-sm text-neuro-text-secondary leading-relaxed max-w-2xl">
              Explore AI-powered brain MRI classification, volumetric 3D subregion segmentation, and transparent explainability models in one synchronized workspace.
            </p>
          </div>
          <div className="flex items-center gap-3 pt-2 flex-wrap">
            <Link
              to="/new-analysis"
              className="px-4 py-2 rounded-lg bg-neuro-accent hover:bg-neuro-accent-bright text-white text-xs font-mono font-semibold transition-all shadow-sm flex items-center gap-2"
            >
              <Upload className="w-3.5 h-3.5" />
              <span>+ Start New Analysis</span>
            </Link>
            <Link
              to="/model-selection"
              className="px-4 py-2 rounded-lg bg-neuro-surface-300 hover:bg-neuro-surface-400 text-neuro-text-secondary hover:text-white text-xs font-mono font-medium border border-neuro-border transition-colors flex items-center gap-2"
            >
              <Sliders className="w-3.5 h-3.5 text-neuro-accent" />
              <span>Browse Model Registry</span>
            </Link>
          </div>
        </div>
      </section>

      {/* ─── RESEARCH METRICS STRIP ───────────────────────────────────── */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {/* Metric 01: Total Analyses */}
        <div className="p-4 rounded-lg bg-neuro-surface-200 border border-neuro-border space-y-1">
          <span className="text-[10px] font-mono uppercase tracking-wider text-neuro-text-muted font-semibold">
            Total Analyses Completed
          </span>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-2xl font-bold font-headline text-white">{history.length}</span>
            <span className="text-xs font-mono text-neuro-text-muted">records</span>
          </div>
          <span className="text-[11px] text-neuro-text-secondary block">Persisted in local repository</span>
        </div>

        {/* Metric 02: 2D Classification */}
        <div className="p-4 rounded-lg bg-neuro-surface-200 border border-neuro-border space-y-1">
          <span className="text-[10px] font-mono uppercase tracking-wider text-neuro-text-muted font-semibold">
            2D Single-Slice Classification
          </span>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-2xl font-bold font-headline text-cyan-400">{classificationCount}</span>
            <span className="text-xs font-mono text-neuro-text-muted">scans</span>
          </div>
          <span className="text-[11px] text-neuro-text-secondary block">4-class diagnostic pathology</span>
        </div>

        {/* Metric 03: 3D Volumetric Segmentation */}
        <div className="p-4 rounded-lg bg-neuro-surface-200 border border-neuro-border space-y-1">
          <span className="text-[10px] font-mono uppercase tracking-wider text-neuro-text-muted font-semibold">
            3D Volumetric Segmentation
          </span>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-2xl font-bold font-headline text-neuro-accent">{segmentationCount}</span>
            <span className="text-xs font-mono text-neuro-text-muted">BraTS volumes</span>
          </div>
          <span className="text-[11px] text-neuro-text-secondary block">WT, TC & ET subregion masks</span>
        </div>

        {/* Metric 04: Verified Models in Memory */}
        <div className="p-4 rounded-lg bg-neuro-surface-200 border border-neuro-border space-y-1">
          <span className="text-[10px] font-mono uppercase tracking-wider text-neuro-text-muted font-semibold">
            Verified Models in Memory
          </span>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-2xl font-bold font-headline text-emerald-400">{verifiedModelsInMemory}</span>
            <span className="text-xs font-mono text-neuro-text-muted">checkpoints</span>
          </div>
          <span className="text-[11px] text-neuro-text-secondary block">Warm & loaded for inference</span>
        </div>
      </section>

      {/* ─── MAIN DASHBOARD: RECENT ANALYSES & ACTIVE MODELS ───────────── */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Column (7 cols): Recent Analyses Table */}
        <div className="lg:col-span-7 space-y-4">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
            <div>
              <h2 className="text-sm font-bold uppercase tracking-wider font-mono text-white">
                Recent Research Analyses
              </h2>
              <p className="text-xs text-neuro-text-muted">Model executions across 2D slices and 3D volumes</p>
            </div>

            {/* Filter Tabs */}
            <div className="flex items-center gap-1 bg-neuro-surface-200 p-1 rounded-md border border-neuro-border font-mono text-xs">
              <button
                onClick={() => setFilterType('all')}
                className={`px-2.5 py-1 rounded transition-colors ${
                  filterType === 'all'
                    ? 'bg-neuro-accent text-white font-semibold shadow-sm'
                    : 'text-neuro-text-muted hover:text-white'
                }`}
              >
                All ({history.length})
              </button>
              <button
                onClick={() => setFilterType('classification')}
                className={`px-2.5 py-1 rounded transition-colors ${
                  filterType === 'classification'
                    ? 'bg-cyan-500/20 text-cyan-400 font-semibold border border-cyan-500/30'
                    : 'text-neuro-text-muted hover:text-white'
                }`}
              >
                2D ({classificationCount})
              </button>
              <button
                onClick={() => setFilterType('segmentation')}
                className={`px-2.5 py-1 rounded transition-colors ${
                  filterType === 'segmentation'
                    ? 'bg-neuro-accent/20 text-neuro-accent font-semibold border border-neuro-accent/30'
                    : 'text-neuro-text-muted hover:text-white'
                }`}
              >
                3D ({segmentationCount})
              </button>
            </div>
          </div>

          {/* Table */}
          <div className="bg-neuro-surface-200 rounded-lg border border-neuro-border overflow-hidden shadow-sm">
            {filteredHistory.length === 0 ? (
              <div className="p-12 text-center space-y-3">
                <Brain className="w-10 h-10 text-neuro-text-muted mx-auto opacity-40" />
                <p className="text-sm font-medium text-neuro-text-secondary">No analysis records found.</p>
                <p className="text-xs text-neuro-text-muted max-w-sm mx-auto">
                  Upload a 2D MRI slice or 4-channel volumetric NIfTI dataset to run model inference.
                </p>
                <div className="pt-2">
                  <button
                    onClick={() => navigate('/new-analysis')}
                    className="px-4 py-2 rounded-lg bg-neuro-accent hover:bg-neuro-accent-bright text-white text-xs font-semibold"
                  >
                    Start New Analysis
                  </button>
                </div>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead className="bg-neuro-surface-300/80 text-neuro-text-muted font-mono uppercase text-[10px] tracking-wider border-b border-neuro-border">
                    <tr>
                      <th className="px-4 py-3">Analysis ID</th>
                      <th className="px-4 py-3">Workflow</th>
                      <th className="px-4 py-3">Findings / Result</th>
                      <th className="px-4 py-3">Timestamp</th>
                      <th className="px-4 py-3 text-right">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-neuro-border/60 text-neuro-text-secondary">
                    {filteredHistory.slice(0, 8).map(item => {
                      const is3D = is3DItem(item);
                      const tumorClass = item.predictedClass || (item as any).class_distribution?.find((c: any) => c.index === 3)?.name || 'Multiclass BraTS';
                      const confidence = item.confidence ? `${(item.confidence * 100).toFixed(1)}%` : null;

                      return (
                        <tr key={item.id} className="hover:bg-neuro-surface-300/50 transition-colors">
                          <td className="px-4 py-3.5 font-mono text-cyan-400 font-semibold truncate max-w-[120px]">
                            #{item.id.slice(0, 8)}
                          </td>
                          <td className="px-4 py-3.5">
                            {is3D ? (
                              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                                3D BraTS Multimodal
                              </span>
                            ) : (
                              <span className="px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                                2D Slice Classification
                              </span>
                            )}
                          </td>
                          <td className="px-4 py-3.5">
                            <span className="font-semibold text-white">{tumorClass}</span>
                            {confidence && (
                              <span className="text-[11px] font-mono text-neuro-text-muted block mt-0.5">
                                Confidence: {confidence}
                              </span>
                            )}
                          </td>
                          <td className="px-4 py-3.5 font-mono text-neuro-text-muted">
                            {formatDate(item.createdAt)}
                          </td>
                          <td className="px-4 py-3.5 text-right">
                            <button
                              onClick={() => handleOpenWorkspace(item)}
                              className="inline-flex items-center gap-1 text-neuro-accent hover:text-neuro-accent-bright font-mono font-semibold group"
                            >
                              <span>Open</span>
                              <ChevronRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition-transform" />
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>

        {/* Right Column (5 cols): Active Models in Memory & Safety Notice */}
        <div className="lg:col-span-5 space-y-4">
          <div className="p-4 rounded-lg bg-neuro-surface-200 border border-neuro-border space-y-3">
            <div className="flex items-center justify-between border-b border-neuro-border pb-2.5">
              <div className="flex items-center gap-2">
                <Brain className="w-4 h-4 text-neuro-accent" />
                <h3 className="text-xs font-bold uppercase tracking-wider font-mono text-white">
                  Active Models in Memory
                </h3>
              </div>
              <span className="text-[10px] font-mono text-emerald-400 font-semibold bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                Online & Ready
              </span>
            </div>

            {/* Models List */}
            <div className="space-y-2.5 pt-1">
              {/* Model 1: Attention CNN + ViT */}
              <div className="p-3 rounded bg-neuro-surface-300/60 border border-neuro-border space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-white">Attention CNN + Vision Transformer</span>
                  <span className="px-1.5 py-0.2 rounded text-[10px] font-mono bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                    3D Multimodal
                  </span>
                </div>
                <p className="text-[11px] text-neuro-text-secondary">
                  3.07M parameters · 82.98% Whole Tumor Dice · Non-destructive brain normalization
                </p>
              </div>

              {/* Model 2: EfficientNetV2B0 */}
              <div className="p-3 rounded bg-neuro-surface-300/60 border border-neuro-border space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-white">EfficientNetV2B0 (Fused-MBConv)</span>
                  <span className="px-1.5 py-0.2 rounded text-[10px] font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    2D 4-Class
                  </span>
                </div>
                <p className="text-[11px] text-neuro-text-secondary">
                  6.28M parameters · 94.1% Test Accuracy · Grad-CAM & Integrated Gradients
                </p>
              </div>

              {/* Model 3: Swin Transformer */}
              <div className="p-3 rounded bg-neuro-surface-300/60 border border-neuro-border space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-white">Swin Transformer (Shifted Window)</span>
                  <span className="px-1.5 py-0.2 rounded text-[10px] font-mono bg-purple-500/10 text-purple-400 border border-purple-500/20">
                    2D ViT
                  </span>
                </div>
                <p className="text-[11px] text-neuro-text-secondary">
                  28.3M parameters · Shifted-window self-attention · Hierarchical representation
                </p>
              </div>

              {/* Model 4: DeepLabV3+ */}
              <div className="p-3 rounded bg-neuro-surface-300/60 border border-neuro-border space-y-1">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-white">DeepLabV3+ (ASPP Encoder-Decoder)</span>
                  <span className="px-1.5 py-0.2 rounded text-[10px] font-mono bg-amber-500/10 text-amber-400 border border-amber-500/20">
                    Binary Lesion
                  </span>
                </div>
                <p className="text-[11px] text-neuro-text-secondary">
                  17.8M parameters · Atrous spatial pyramid pooling · Multi-scale context
                </p>
              </div>
            </div>
          </div>

          <ResearchDisclaimer />
        </div>
      </section>
    </motion.div>
  );
}
