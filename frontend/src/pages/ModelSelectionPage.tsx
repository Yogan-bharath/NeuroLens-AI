import { useState, useMemo } from 'react';
import { motion } from 'framer-motion';
import { Link, useNavigate } from 'react-router-dom';
import { toast } from 'sonner';
import {
  Brain,
  Sliders,
  CheckCircle2,
  ShieldCheck,
  Activity,
  Layers,
  Sparkles,
  ArrowRight,
  Cpu,
  Clock,
  FileCheck2,
  Check,
  Zap,
} from 'lucide-react';
import PageHeader from '@/components/layout/PageHeader';
import { ResearchDisclaimer } from '@/components/common/ResearchDisclaimer';
import {
  classificationModelsRegistry,
  volumetricSegmentationModelsRegistry,
  binarySegmentationModelsRegistry,
} from '@/services/modelRegistry';
import { useConfiguration } from '@/context/ConfigurationContext';

type TabType = 'all' | 'classification' | 'volumetric' | 'binary';

export default function ModelSelectionPage() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<TabType>('all');

  const {
    selectedClassificationModelId,
    selectedSegmentationModelId,
    setClassificationModel,
    setSegmentationModel,
  } = useConfiguration();

  // Combine models into typed registry items
  const allModels = useMemo(() => {
    const classModels = classificationModelsRegistry
      .filter(m => m.status === 'available')
      .map(m => ({
        ...m,
        category: 'classification' as const,
        categoryLabel: '2D Classification',
        isSelected: m.id === selectedClassificationModelId,
      }));

    const volModels = volumetricSegmentationModelsRegistry
      .filter(m => m.status === 'available')
      .map(m => ({
        ...m,
        category: 'volumetric' as const,
        categoryLabel: '3D Volumetric Segmentation',
        isSelected: m.id === selectedSegmentationModelId,
      }));

    const binModels = binarySegmentationModelsRegistry
      .filter(m => m.status === 'available')
      .map(m => ({
        ...m,
        category: 'binary' as const,
        categoryLabel: '2D Lesion Segmentation',
        isSelected: m.id === selectedSegmentationModelId,
      }));

    return [...classModels, ...volModels, ...binModels];
  }, [selectedClassificationModelId, selectedSegmentationModelId]);

  const filteredModels = useMemo(() => {
    if (activeTab === 'all') return allModels;
    return allModels.filter(m => m.category === activeTab);
  }, [allModels, activeTab]);

  const activeClassModel = classificationModelsRegistry.find(m => m.id === selectedClassificationModelId);
  const activeSegModel = volumetricSegmentationModelsRegistry.find(m => m.id === selectedSegmentationModelId) ||
    binarySegmentationModelsRegistry.find(m => m.id === selectedSegmentationModelId);

  const handleSelectModel = (model: any) => {
    if (model.category === 'classification') {
      setClassificationModel(model.id);
      toast.success(`Active 2D Classification model set to ${model.name}.`);
    } else {
      setSegmentationModel(model.id);
      toast.success(`Active Segmentation model set to ${model.name}.`);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className="space-y-6 pb-12 max-w-6xl mx-auto text-neuro-text-primary"
    >
      <PageHeader
        title="Model Registry & Architecture Catalog"
        subtitle="Explore verified neural architectures, inspect benchmark evaluations, and select target models for subsequent inference workflows."
      >
        <Link
          to="/new-analysis"
          className="px-4 py-2 rounded-lg bg-neuro-accent hover:bg-neuro-accent-bright text-white text-xs font-mono font-semibold transition-all shadow-sm flex items-center gap-2"
        >
          <span>Run Analysis</span>
          <ArrowRight className="w-3.5 h-3.5" />
        </Link>
      </PageHeader>

      {/* ─── ACTIVE MODEL SELECTION BANNER ─────────────────────────────── */}
      <div className="p-4 rounded-xl bg-neuro-surface-200 border border-neuro-border flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-neuro-accent/10 border border-neuro-accent/20 flex items-center justify-center text-neuro-accent flex-shrink-0">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <span>Active Production Pipeline Models</span>
              <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                PERSISTED
              </span>
            </h3>
            <div className="text-xs text-neuro-text-secondary mt-1 flex flex-wrap items-center gap-3">
              <div>
                2D Classification:{' '}
                <strong className="text-emerald-400 font-semibold font-mono">
                  {activeClassModel?.name || selectedClassificationModelId}
                </strong>
              </div>
              <span className="text-neuro-text-muted hidden sm:inline">•</span>
              <div>
                3D Volumetric Segmentation:{' '}
                <strong className="text-cyan-400 font-semibold font-mono">
                  {activeSegModel?.name || selectedSegmentationModelId}
                </strong>
              </div>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2 w-full md:w-auto justify-end">
          <Link
            to="/model-performance"
            className="px-3 py-1.5 rounded-lg bg-neuro-surface-300 hover:bg-neuro-surface-400 text-neuro-text-secondary hover:text-white text-xs font-mono font-medium border border-neuro-border transition-colors whitespace-nowrap"
          >
            Compare Metrics →
          </Link>
          <Link
            to="/new-analysis"
            className="px-3.5 py-1.5 rounded-lg bg-neuro-accent hover:bg-neuro-accent-bright text-white text-xs font-mono font-semibold transition-colors flex items-center gap-1.5 whitespace-nowrap"
          >
            <Zap className="w-3.5 h-3.5" />
            <span>Launch Analysis</span>
          </Link>
        </div>
      </div>

      {/* ─── FILTER TABS ─────────────────────────────────────────────── */}
      <div className="flex items-center justify-between border-b border-neuro-border pb-3 flex-wrap gap-2">
        <div className="flex items-center gap-1.5 bg-neuro-surface-200 p-1 rounded-lg border border-neuro-border font-mono text-xs flex-wrap">
          <button
            onClick={() => setActiveTab('all')}
            className={`px-3 py-1.5 rounded transition-all ${
              activeTab === 'all'
                ? 'bg-neuro-accent text-white font-semibold shadow-sm'
                : 'text-neuro-text-muted hover:text-white'
            }`}
          >
            All Architectures ({allModels.length})
          </button>
          <button
            onClick={() => setActiveTab('classification')}
            className={`px-3 py-1.5 rounded transition-all ${
              activeTab === 'classification'
                ? 'bg-emerald-500/20 text-emerald-400 font-semibold border border-emerald-500/30'
                : 'text-neuro-text-muted hover:text-white'
            }`}
          >
            2D Classification ({allModels.filter(m => m.category === 'classification').length})
          </button>
          <button
            onClick={() => setActiveTab('volumetric')}
            className={`px-3 py-1.5 rounded transition-all ${
              activeTab === 'volumetric'
                ? 'bg-cyan-500/20 text-cyan-400 font-semibold border border-cyan-500/30'
                : 'text-neuro-text-muted hover:text-white'
            }`}
          >
            3D Volumetric ({allModels.filter(m => m.category === 'volumetric').length})
          </button>
          <button
            onClick={() => setActiveTab('binary')}
            className={`px-3 py-1.5 rounded transition-all ${
              activeTab === 'binary'
                ? 'bg-purple-500/20 text-purple-400 font-semibold border border-purple-500/30'
                : 'text-neuro-text-muted hover:text-white'
            }`}
          >
            2D Segmentation ({allModels.filter(m => m.category === 'binary').length})
          </button>
        </div>

        <span className="text-xs font-mono text-neuro-text-muted">
          Framework: TensorFlow 2.18 / Keras 3
        </span>
      </div>

      {/* ─── MODEL CARDS GRID ─────────────────────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {filteredModels.map(model => {
          const isSelected = model.isSelected;
          const isVerified = model.status === 'available';

          const getCategoryBadgeClass = () => {
            if (model.category === 'classification') return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20';
            if (model.category === 'volumetric') return 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20';
            return 'bg-purple-500/10 text-purple-400 border-purple-500/20';
          };

          return (
            <div
              key={model.id}
              className={`p-5 rounded-xl border transition-all space-y-4 flex flex-col justify-between ${
                isSelected
                  ? 'bg-neuro-surface-200/90 border-neuro-accent shadow-lg ring-1 ring-neuro-accent/40'
                  : 'bg-neuro-surface-200/50 border-neuro-border hover:border-neuro-border/80'
              }`}
            >
              <div className="space-y-3">
                {/* Top Badge Row */}
                <div className="flex items-center justify-between gap-2 flex-wrap">
                  <div className="flex items-center gap-2">
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase tracking-wider border ${getCategoryBadgeClass()}`}
                    >
                      {model.categoryLabel}
                    </span>
                    {isSelected && (
                      <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 flex items-center gap-1">
                        <CheckCircle2 className="w-3 h-3" />
                        Selected
                      </span>
                    )}
                  </div>
                  <span
                    className={`inline-flex items-center gap-1.5 text-xs font-mono ${
                      isVerified ? 'text-emerald-400' : 'text-amber-400'
                    }`}
                  >
                    <span
                      className={`w-1.5 h-1.5 rounded-full ${
                        isVerified ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'
                      }`}
                    />
                    {isVerified ? 'Verified & Loaded' : 'Candidate'}
                  </span>
                </div>

                {/* Title & Architecture */}
                <div>
                  <h3 className="text-base font-bold text-white flex items-center justify-between">
                    <span>{model.name}</span>
                    <span className="text-xs font-mono text-neuro-text-muted font-normal">ID: {model.id}</span>
                  </h3>
                  <p className="text-xs font-mono text-neuro-text-muted mt-0.5">{model.architecture}</p>
                </div>

                {/* Description */}
                <p className="text-xs text-neuro-text-secondary leading-relaxed">{model.description}</p>

                {/* Metrics / Parameters Specs */}
                <div className="grid grid-cols-2 gap-2 pt-2 border-t border-neuro-border/60 text-xs font-mono">
                  <div className="p-2 rounded bg-neuro-surface-300/60 border border-neuro-border">
                    <span className="text-[10px] text-neuro-text-muted uppercase block">Parameters</span>
                    <span className="font-bold text-white">{model.parametersCount || 'Artifact-defined'}</span>
                  </div>
                  <div className="p-2 rounded bg-neuro-surface-300/60 border border-neuro-border">
                    <span className="text-[10px] text-neuro-text-muted uppercase block">Version / Tag</span>
                    <span className="font-bold text-cyan-400">{model.version}</span>
                  </div>
                </div>

                {/* Benchmark Ground-Truth Metrics */}
                {model.metrics && (
                  <div className="p-3 rounded bg-neuro-surface-300/40 border border-neuro-border/60 space-y-1.5 text-xs">
                    <span className="text-[10px] font-mono text-neuro-text-muted uppercase tracking-wider font-semibold block">
                      Evaluated Benchmark: {model.metrics.evaluationDataset}
                    </span>
                    <div className="flex items-center gap-4 font-mono text-xs flex-wrap">
                      {'testAccuracy' in model.metrics && (
                        <div>
                          <span className="text-neuro-text-muted text-[11px]">Accuracy: </span>
                          <strong className="text-emerald-400 font-bold">
                            {((model.metrics as any).testAccuracy * 100).toFixed(1)}%
                          </strong>
                        </div>
                      )}
                      {'diceCoefficient' in model.metrics && (
                        <div>
                          <span className="text-neuro-text-muted text-[11px]">Dice: </span>
                          <strong className="text-emerald-400 font-bold">
                            {((model.metrics as any).diceCoefficient * 100).toFixed(1)}%
                          </strong>
                        </div>
                      )}
                      {'iouScore' in model.metrics && (
                        <div>
                          <span className="text-neuro-text-muted text-[11px]">IoU: </span>
                          <strong className="text-cyan-400 font-bold">
                            {((model.metrics as any).iouScore * 100).toFixed(1)}%
                          </strong>
                        </div>
                      )}
                      <div>
                        <span className="text-neuro-text-muted text-[11px]">Latency: </span>
                        <strong className="text-neuro-text-secondary font-bold">
                          {model.metrics.inferenceTimeMs} ms
                        </strong>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Action Buttons: Select Model Button */}
              <div className="pt-3 border-t border-neuro-border/60 flex items-center justify-between gap-3">
                <span className="text-[11px] font-mono text-neuro-text-muted truncate max-w-[200px]">
                  {model.supportedXaiMethods?.join(' · ').toUpperCase() || 'GRAD-CAM'}
                </span>

                <div className="flex items-center gap-2">
                  {isSelected ? (
                    <button
                      type="button"
                      disabled
                      className="px-3.5 py-1.5 rounded-lg bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 text-xs font-mono font-semibold flex items-center gap-1.5 cursor-default shadow-sm"
                    >
                      <Check className="w-3.5 h-3.5" />
                      <span>Active Model</span>
                    </button>
                  ) : (
                    <button
                      type="button"
                      onClick={() => handleSelectModel(model)}
                      className="px-3.5 py-1.5 rounded-lg bg-neuro-accent hover:bg-neuro-accent-bright text-white text-xs font-mono font-semibold transition-all shadow-sm flex items-center gap-1.5 cursor-pointer"
                    >
                      <Sparkles className="w-3.5 h-3.5" />
                      <span>Select Model</span>
                    </button>
                  )}

                  <Link
                    to="/new-analysis"
                    className="p-1.5 rounded-lg bg-neuro-surface-300 hover:bg-neuro-surface-400 border border-neuro-border text-neuro-text-muted hover:text-white transition-colors"
                    title="Run Analysis with this model"
                  >
                    <ArrowRight className="w-4 h-4" />
                  </Link>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      <ResearchDisclaimer />
    </motion.div>
  );
}
