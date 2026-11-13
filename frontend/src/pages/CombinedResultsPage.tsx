import { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  ArrowLeft,
  Brain,
  Layers,
  Sparkles,
  Download,
  ChevronRight,
  ShieldCheck,
  RefreshCw,
  Activity,
  Sliders,
  AlertTriangle,
  CheckCircle2,
  AlertCircle,
  Clock,
  Eye,
} from 'lucide-react';
import { LoadingSpinner } from '@/components/common/LoadingState';
import { ResearchDisclaimer } from '@/components/common/ResearchDisclaimer';
import Tumor3DViewer from '@/components/viewer/Tumor3DViewer';
import { getAnalysisById } from '@/services/api';
import ExportReportButton from '@/components/common/ExportReportButton';
import { formatDate, formatPercentage } from '@/lib/utils';

export default function CombinedResultsPage() {
  const { analysisId } = useParams<{ analysisId: string }>();
  const navigate = useNavigate();

  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<any>(null);

  // 2D Viewport State
  const [showHeatmap, setShowHeatmap] = useState<boolean>(true);
  const [camOpacity, setCamOpacity] = useState<number>(0.65);

  // Active Tab for detailed inspection
  const [activeTab, setActiveTab] = useState<'both' | 'classification' | 'segmentation'>('both');

  useEffect(() => {
    if (!analysisId) {
      setError('Analysis ID is required.');
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    setError(null);

    getAnalysisById(analysisId)
      .then(res => {
        if (!res) {
          setError('Combined Analysis record not found.');
          return;
        }
        setData(res);
      })
      .catch(err => {
        setError(err instanceof Error ? err.message : 'Failed to retrieve combined analysis.');
      })
      .finally(() => {
        setIsLoading(false);
      });
  }, [analysisId]);

  if (isLoading) {
    return (
      <div className="min-h-[500px] flex items-center justify-center">
        <LoadingSpinner text="Retrieving Combined 2D + 3D Multimodal Dossier..." />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="max-w-2xl mx-auto py-12 px-4 space-y-4 text-center">
        <div className="p-8 rounded-xl bg-neuro-surface-200 border border-neuro-border space-y-4">
          <AlertCircle className="w-10 h-10 text-red-400 mx-auto" />
          <h2 className="text-lg font-bold text-white">Unable to Load Combined Analysis</h2>
          <p className="text-sm text-neuro-text-secondary">{error || 'Record does not exist.'}</p>
          <div className="pt-2 flex justify-center gap-3">
            <button
              onClick={() => navigate('/new-analysis')}
              className="px-4 py-2 rounded-lg bg-neuro-accent hover:bg-neuro-accent-bright text-white text-xs font-semibold"
            >
              Start New Analysis
            </button>
            <button
              onClick={() => navigate('/history')}
              className="px-4 py-2 rounded-lg bg-neuro-surface-300 text-neuro-text-secondary text-xs font-medium border border-neuro-border"
            >
              History
            </button>
          </div>
        </div>
      </div>
    );
  }

  const cls = data.classification || {};
  const seg = data.segmentation || {};
  const errors = data.errors || {};

  const hasCls = Boolean(data.classification_id && !errors.classification);
  const hasSeg = Boolean(data.segmentation_id && !errors.segmentation);

  const topPrediction = cls.top_prediction || cls.predictedClass || data.predictedClass || 'Glioma';
  const confidence = cls.confidence ?? data.confidence ?? 0.0;
  const scores: Array<any> = cls.scores || [];

  const originalImage = cls.image_url || cls.imageDataUrl || data.imageDataUrl || data.thumbnailUrl;
  const gradcamOverlay = cls.gradcam_overlay || cls.gradcamOverlay || originalImage;

  const surface3D = seg.surface_3d || {
    bounding_box: { min: [0, 0, 0], max: [1, 1, 1], center: [0.5, 0.5, 0.5] },
    classes: [],
    points: {},
  };
  const dimensions: [number, number, number] = seg.volume_dimensions || [240, 240, 155];
  const classDistribution: Array<any> = seg.class_distribution || [];
  const totalTumorVolumeMm3 = classDistribution
    .filter(c => c.index > 0)
    .reduce((acc, c) => acc + (c.volume_mm3 || 0), 0);
  const totalTumorVolumeCm3 = (totalTumorVolumeMm3 / 1000).toFixed(2);

  const downloadUrl = seg.download_url || `/api/analyses/${data.segmentation_id || analysisId}/segmentation`;

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className="space-y-4 pb-12 max-w-[1520px] mx-auto text-neuro-text-primary"
    >
      {/* ─── HEADER BAR ──────────────────────────────────────────────── */}
      <header className="flex flex-col lg:flex-row items-start lg:items-center justify-between gap-3 border-b border-neuro-border pb-3 bg-neuro-surface-100/60 p-3 rounded-lg">
        <div className="flex items-center gap-2 text-xs flex-wrap">
          <Link to="/new-analysis" className="text-neuro-text-muted hover:text-neuro-accent flex items-center gap-1 transition-colors">
            <ArrowLeft className="w-3.5 h-3.5" /> Analysis Hub
          </Link>
          <ChevronRight className="w-3 h-3 text-neuro-text-muted" />
          <Link to="/history" className="text-neuro-text-muted hover:text-neuro-accent transition-colors">
            History
          </Link>
          <ChevronRight className="w-3 h-3 text-neuro-text-muted" />
          <span className="font-mono text-cyan-400 font-semibold">{analysisId}</span>
          <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-purple-500/10 text-purple-400 border border-purple-500/20 font-mono">
            COMBINED 2D + 3D STUDY
          </span>
          <span className="text-neuro-text-muted">•</span>
          <span className="text-[11px] text-neuro-text-secondary font-mono">
            {hasCls && hasSeg ? 'Ensemble Synchronized' : 'Partial Execution'}
          </span>
        </div>

        <div className="flex items-center gap-2 flex-wrap w-full lg:w-auto justify-end">
          {hasSeg && (
            <a
              href={downloadUrl}
              download
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-neuro-surface-200 hover:bg-neuro-surface-300 border border-neuro-border text-xs font-mono transition-colors"
            >
              <Download className="w-3.5 h-3.5 text-neuro-accent" />
              <span>Download 3D NIfTI</span>
            </a>
          )}
          <ExportReportButton analysis={data} />
          <button
            onClick={() => navigate('/new-analysis')}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-md bg-neuro-accent hover:bg-neuro-accent-bright text-white font-semibold text-xs transition-colors shadow-sm"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>New Analysis</span>
          </button>
        </div>
      </header>

      {/* ─── ERROR ALERTS FOR PARTIAL FAILURES ─────────────────────────── */}
      {errors.classification && (
        <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/30 flex items-center gap-3 text-xs text-red-300">
          <AlertCircle className="w-4 h-4 flex-shrink-0 text-red-400" />
          <span><strong>2D Classification Note:</strong> {errors.classification}</span>
        </div>
      )}
      {errors.segmentation && (
        <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/30 flex items-center gap-3 text-xs text-amber-300">
          <AlertTriangle className="w-4 h-4 flex-shrink-0 text-amber-400" />
          <span><strong>3D Segmentation Note:</strong> {errors.segmentation}</span>
        </div>
      )}

      {/* ─── TOP EXECUTIVE SUMMARY BANNER ─────────────────────────────── */}
      <section className="p-5 rounded-xl bg-neuro-surface-200 border border-neuro-border grid grid-cols-1 md:grid-cols-3 gap-4 items-center">
        {/* Finding 1: 2D Classification */}
        <div className="flex items-center gap-3 border-b md:border-b-0 md:border-r border-neuro-border/60 pb-3 md:pb-0 md:pr-4">
          <div className="w-10 h-10 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400 flex-shrink-0">
            <Brain className="w-5 h-5" />
          </div>
          <div>
            <span className="text-[10px] font-mono uppercase text-neuro-text-muted block">
              2D Pathology (EfficientNetV2B0)
            </span>
            <div className="flex items-center gap-2 mt-0.5">
              <span className="text-base font-bold text-white capitalize">{topPrediction}</span>
              {hasCls && (
                <span className="px-2 py-0.2 rounded text-[10px] font-mono font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                  {formatPercentage(confidence)}
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Finding 2: 3D Volumetrics */}
        <div className="flex items-center gap-3 border-b md:border-b-0 md:border-r border-neuro-border/60 pb-3 md:pb-0 md:pr-4">
          <div className="w-10 h-10 rounded-lg bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center text-cyan-400 flex-shrink-0">
            <Layers className="w-5 h-5" />
          </div>
          <div>
            <span className="text-[10px] font-mono uppercase text-neuro-text-muted block">
              3D Tumor Mass (Attention-ViT)
            </span>
            <div className="flex items-baseline gap-2 mt-0.5">
              <span className="text-base font-bold font-mono text-cyan-400">{totalTumorVolumeCm3}</span>
              <span className="text-xs font-mono text-neuro-text-muted">cm³</span>
            </div>
          </div>
        </div>

        {/* Study Timestamp & Direct Links */}
        <div className="flex items-center justify-between gap-2">
          <div className="space-y-0.5">
            <span className="text-[10px] font-mono uppercase text-neuro-text-muted block">Study Registered</span>
            <span className="text-xs font-mono text-neuro-text-secondary">{formatDate(data.createdAt || data.created_at)}</span>
          </div>
          <div className="flex items-center gap-1.5 font-mono text-xs">
            {data.classification_id && (
              <Link
                to={`/analysis/${data.classification_id}/2d-results`}
                className="px-2.5 py-1 rounded bg-neuro-surface-300 hover:bg-neuro-surface-400 text-emerald-400 border border-neuro-border transition-colors text-[11px]"
              >
                2D Results →
              </Link>
            )}
            {data.segmentation_id && (
              <Link
                to={`/analysis/${data.segmentation_id}/3d-results`}
                className="px-2.5 py-1 rounded bg-neuro-surface-300 hover:bg-neuro-surface-400 text-cyan-400 border border-neuro-border transition-colors text-[11px]"
              >
                3D Results →
              </Link>
            )}
          </div>
        </div>
      </section>

      {/* ─── DUAL-STREAM WORKSPACE GRID ───────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* LEFT COLUMN: 2D CLASSIFICATION & SALIENCY */}
        <div className="space-y-4">
          <div className="p-4 rounded-xl bg-neuro-surface-200 border border-neuro-border space-y-3">
            <div className="flex items-center justify-between border-b border-neuro-border pb-3">
              <div className="flex items-center gap-2">
                <Brain className="w-4 h-4 text-emerald-400" />
                <h3 className="text-xs font-bold uppercase tracking-wider font-mono text-white">
                  2D MRI Slice & Grad-CAM Heatmap
                </h3>
              </div>
              <button
                type="button"
                onClick={() => setShowHeatmap(!showHeatmap)}
                className={`px-2 py-0.5 rounded text-xs font-mono border transition-all ${
                  showHeatmap
                    ? 'bg-neuro-accent/20 border-neuro-accent text-neuro-accent'
                    : 'bg-neuro-surface-300 border-neuro-border text-neuro-text-muted'
                }`}
              >
                {showHeatmap ? 'Heatmap ON' : 'Heatmap OFF'}
              </button>
            </div>

            {/* Image Viewer */}
            <div className="relative aspect-square max-h-[380px] w-full rounded-lg bg-black overflow-hidden border border-neuro-border flex items-center justify-center">
              {originalImage ? (
                <>
                  <img
                    src={originalImage}
                    alt="2D MRI Slice"
                    className="absolute inset-0 w-full h-full object-contain"
                  />
                  {showHeatmap && gradcamOverlay && (
                    <img
                      src={gradcamOverlay}
                      alt="Grad-CAM Overlay"
                      className="absolute inset-0 w-full h-full object-contain pointer-events-none mix-blend-screen transition-opacity"
                      style={{ opacity: camOpacity }}
                    />
                  )}
                </>
              ) : (
                <div className="text-center text-neuro-text-muted p-6">
                  <Brain className="w-10 h-10 mx-auto opacity-30 mb-2" />
                  <p className="text-xs">No 2D slice available</p>
                </div>
              )}
            </div>

            {/* Opacity slider */}
            {showHeatmap && (
              <div className="flex items-center justify-between gap-4 pt-1 text-xs font-mono">
                <span className="text-neuro-text-muted">Heatmap Opacity:</span>
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    min="0.1"
                    max="1.0"
                    step="0.05"
                    value={camOpacity}
                    onChange={e => setCamOpacity(parseFloat(e.target.value))}
                    className="w-28 accent-neuro-accent cursor-pointer"
                  />
                  <span className="w-8 text-right font-bold text-white">{Math.round(camOpacity * 100)}%</span>
                </div>
              </div>
            )}
          </div>

          {/* Probabilities */}
          <div className="p-4 rounded-xl bg-neuro-surface-200 border border-neuro-border space-y-3">
            <h4 className="text-xs font-bold uppercase tracking-wider font-mono text-white">
              Probability Distribution
            </h4>
            <div className="space-y-2 pt-1 text-xs">
              {scores.map((s: any) => {
                const isTop = s.label.toLowerCase() === topPrediction.toLowerCase();
                const pct = s.percentage ?? Math.round(s.score * 100);
                return (
                  <div key={s.label} className="space-y-1">
                    <div className="flex justify-between">
                      <span className={`capitalize ${isTop ? 'text-white font-bold' : 'text-neuro-text-secondary'}`}>
                        {s.label.replace(/_/g, ' ')}
                      </span>
                      <span className="font-mono text-neuro-text-muted">{pct.toFixed(1)}%</span>
                    </div>
                    <div className="h-1.5 w-full bg-neuro-surface-400 rounded-full overflow-hidden">
                      <div
                        className={`h-full rounded-full ${isTop ? 'bg-emerald-400' : 'bg-neuro-text-muted/40'}`}
                        style={{ width: `${Math.max(pct, 1)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* RIGHT COLUMN: 3D VOLUMETRIC SEGMENTATION WORKSTATION */}
        <div className="space-y-4">
          <div className="p-4 rounded-xl bg-neuro-surface-200 border border-neuro-border space-y-3">
            <div className="flex items-center justify-between border-b border-neuro-border pb-3">
              <div className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-cyan-400" />
                <h3 className="text-xs font-bold uppercase tracking-wider font-mono text-white">
                  Interactive 3D Tumor & Brain Canvas
                </h3>
              </div>
              <span className="text-[10px] font-mono text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/20">
                RAY-MARCHED MIP
              </span>
            </div>

            {/* 3D Canvas Container */}
            <div className="w-full bg-[#050a14] rounded-lg border border-neuro-border flex items-center justify-center p-2 min-h-[380px]">
              <Tumor3DViewer
                surfaceData={surface3D}
                dimensions={dimensions}
                anatomicalOrientation={seg.anatomical_orientation || ['R', 'A', 'S']}
              />
            </div>
          </div>

          {/* Subregions Table */}
          <div className="p-4 rounded-xl bg-neuro-surface-200 border border-neuro-border space-y-2">
            <h4 className="text-xs font-bold uppercase tracking-wider font-mono text-white">
              BraTS Subregion Delineation
            </h4>
            <div className="space-y-1.5 pt-1">
              {classDistribution.filter(c => c.index > 0).map(c => (
                <div key={c.index} className="p-2 rounded bg-neuro-surface-300/60 border border-neuro-border flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2">
                    <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: c.color || '#28C7B7' }} />
                    <span className="font-semibold text-white">{c.name}</span>
                  </div>
                  <div className="text-right font-mono">
                    <span className="text-cyan-400 font-bold">{(c.volume_mm3 / 1000).toFixed(2)} cm³</span>
                    <span className="text-[10px] text-neuro-text-muted ml-2">({c.percentage}%)</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <ResearchDisclaimer />
    </motion.div>
  );
}

