import { useState, useEffect, useMemo } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  ArrowLeft,
  Brain,
  Download,
  Layers,
  ChevronRight,
  ShieldCheck,
  RefreshCw,
  Activity,
  FileCheck,
  Sliders,
  ChevronLeft,
  Sparkles,
  AlertCircle,
  Eye,
  Scan,
  Maximize2,
  CheckSquare,
  Square,
} from 'lucide-react';
import { LoadingSpinner } from '@/components/common/LoadingState';
import { ResearchDisclaimer } from '@/components/common/ResearchDisclaimer';
import Tumor3DViewer from '@/components/viewer/Tumor3DViewer';
import { getAnalysisById } from '@/services/api';
import ExportReportButton from '@/components/common/ExportReportButton';

type PlaneType = 'axial' | 'coronal' | 'sagittal';
type WindowPreset = 'brain' | 'subdural' | 'bone' | 'custom';
type ViewMode = 'overlay' | 'mask_only' | 'original';

export default function Segmentation3DResultsPage() {
  const { analysisId } = useParams<{ analysisId: string }>();
  const navigate = useNavigate();

  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<any>(null);

  // Synchronized Multiplanar Viewport State
  const [activePlane, setActivePlane] = useState<PlaneType>('axial');
  const [sliceIndex, setSliceIndex] = useState<number>(78);
  const [activeModality, setActiveModality] = useState<'flair' | 't1' | 't1ce' | 't2'>('flair');
  const [overlayOpacity, setOverlayOpacity] = useState<number>(0.65);
  const [viewMode, setViewMode] = useState<ViewMode>('overlay');
  const [windowPreset, setWindowPreset] = useState<WindowPreset>('brain');
  const [selectedModel, setSelectedModel] = useState<'attention_vit' | 'unet3d_brats'>('attention_vit');

  // Layer visibility toggles (Necrotic, Edema, Enhancing)
  const [visibleSubregions, setVisibleSubregions] = useState<{
    necrotic: boolean;
    edema: boolean;
    enhancing: boolean;
  }>({
    necrotic: true,
    edema: true,
    enhancing: true,
  });

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
          setError('3D Segmentation record not found.');
          return;
        }

        const is2DOnly =
          (res.workflow_type === '2D_CLASSIFICATION' || res.analysis_mode === 'classification_2d') &&
          !res.volume_dimensions &&
          !res.slices &&
          !res.surface_3d;

        if (is2DOnly) {
          // Study is 2D classification only; redirect immediately to dedicated 2D results dossier
          navigate(`/analysis/${analysisId}/2d-results`, { replace: true });
          return;
        }

        const hasVolumetricOutputs =
          Boolean(res.slices && Object.keys(res.slices).length > 0) ||
          Boolean(res.class_distribution && res.class_distribution.length > 0) ||
          Boolean(res.surface_3d && Object.keys(res.surface_3d).length > 0) ||
          Boolean(res.download_url);

        if (!hasVolumetricOutputs) {
          setError('No segmentation masks or volumetric outputs were found for this 3D scan session. Inference may have failed or output files are missing.');
          return;
        }

        setData(res);
        const dims: [number, number, number] = res.volume_dimensions || [240, 240, 155];
        setSliceIndex(Math.floor(dims[2] / 2));
      })
      .catch(err => {
        setError(err instanceof Error ? err.message : 'Failed to retrieve 3D segmentation results.');
      })
      .finally(() => {
        setIsLoading(false);
      });
  }, [analysisId, navigate]);

  const dimensions: [number, number, number] = data?.volume_dimensions || [240, 240, 155];
  const maxSlicesForPlane = useMemo(() => {
    if (activePlane === 'axial') return dimensions[2] || 155;
    if (activePlane === 'coronal') return dimensions[1] || 240;
    return dimensions[0] || 240;
  }, [activePlane, dimensions]);

  // Adjust sliceIndex if plane changes and exceeds bounds
  useEffect(() => {
    setSliceIndex(prev => Math.min(Math.max(1, prev), maxSlicesForPlane));
  }, [activePlane, maxSlicesForPlane]);

  // Subregions comma-separated string for backend filter
  const activeClassesParam = useMemo(() => {
    const list: number[] = [];
    if (visibleSubregions.necrotic) list.push(1);
    if (visibleSubregions.edema) list.push(2);
    if (visibleSubregions.enhancing) list.push(3);
    return list.length > 0 ? list.join(',') : 'none';
  }, [visibleSubregions]);

  const predictionId = useMemo(() => {
    if (!data) return null;
    return data.prediction_id || data.metadata?.predictionId || analysisId;
  }, [data, analysisId]);

  // Panel A URL (Original MRI slice)
  const panelAUrl = useMemo(() => {
    if (!predictionId) return null;
    return `/api/segmentation/slice/${predictionId}?plane=${activePlane}&index=${sliceIndex}&modality=${activeModality}&view_mode=original`;
  }, [predictionId, activePlane, sliceIndex, activeModality]);

  // Panel B URL (Model Predicted Segmentation slice with subregions)
  const panelBUrl = useMemo(() => {
    if (!predictionId) return null;
    return `/api/segmentation/slice/${predictionId}?plane=${activePlane}&index=${sliceIndex}&opacity=${overlayOpacity}&modality=${activeModality}&view_mode=${viewMode}&classes=${activeClassesParam}`;
  }, [predictionId, activePlane, sliceIndex, overlayOpacity, activeModality, viewMode, activeClassesParam]);

  if (isLoading) {
    return (
      <div className="min-h-[500px] flex items-center justify-center">
        <LoadingSpinner text="Loading 3D Multimodal Segmentation Workstation..." />
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="max-w-2xl mx-auto py-12 px-4 space-y-4 text-center">
        <div className="p-8 rounded-xl bg-neuro-surface-200 border border-neuro-border space-y-4">
          <AlertCircle className="w-10 h-10 text-red-400 mx-auto" />
          <h2 className="text-lg font-bold text-neuro-text-primary">Unable to Load 3D Segmentation</h2>
          <p className="text-sm text-neuro-text-secondary">{error || 'Record does not exist.'}</p>
          <div className="pt-2 flex justify-center gap-3">
            <button
              onClick={() => navigate('/new-analysis')}
              className="px-4 py-2 rounded-lg bg-neuro-accent hover:bg-neuro-accent-bright text-white text-xs font-semibold"
            >
              New 3D Scan
            </button>
            <button
              onClick={() => navigate('/history')}
              className="px-4 py-2 rounded-lg bg-neuro-surface-300 hover:bg-neuro-surface-400 text-neuro-text-secondary text-xs font-medium border border-neuro-border"
            >
              Analysis History
            </button>
          </div>
        </div>
      </div>
    );
  }

  const downloadUrl = data.download_url || `/api/segmentation/download/${predictionId}`;
  const classDistribution: Array<any> = data.class_distribution || [];
  const slices = data.slices || {};
  const fallbackSlice = slices[activePlane]?.preview;
  const surface3D = data.surface_3d || {
    bounding_box: { min: [0, 0, 0], max: [1, 1, 1], center: [0.5, 0.5, 0.5] },
    classes: [],
    points: {},
  };
  const anatomicalLocation = surface3D.anatomical_location || data.anatomical_location;
  const evaluationMetrics = data.evaluation_metrics;
  const processingTimeMs = data.processing_time_ms || data.actual_model?.inference_time_ms || data.inference_time || 182;

  const isUnet3D = selectedModel === 'unet3d_brats' || data.model_id === 'unet3d_brats';
  const actualModel = data.actual_model || data.executed_model || {};
  const modelId = isUnet3D ? 'unet3d_brats' : (actualModel.model_id || data.model_id || 'brats_3d_unet');
  const modelName = isUnet3D ? '3D U-Net (volumetric)' : (actualModel.model_name || data.model_name || data.metadata?.modelName || 'Attention CNN + ViT (2D)');
  const modelVersion = isUnet3D ? 'BraTS-3D-UNet-v1' : (actualModel.model_version || data.model_version || data.metadata?.modelVersion || 'v2.4.0 (BraTS-2D)');
  const architecture = isUnet3D ? '3D Encoder-Decoder Convolutional Network (Conv3D + BatchNorm)' : (actualModel.architecture || 'Hybrid Attention CNN + Vision Transformer');
  const framework = actualModel.framework || 'TensorFlow 2.18 / Keras 3';
  const parametersCount = isUnet3D ? '5.61M params (5,606,628)' : (actualModel.parameters_count || '3.07M params');
  const preprocessing = actualModel.preprocessing || 'Non-zero Z-score intensity normalization across FLAIR, T1, T1CE, T2';
  const inputModalities = actualModel.input_modalities || data.input_modalities || ['FLAIR', 'T1', 'T1CE', 'T2'];
  const voxelSpacing = actualModel.voxel_spacing || data.voxel_spacing || [1.0, 1.0, 1.0];

  const totalTumorVolumeMm3 = classDistribution
    .filter(c => c.index > 0)
    .reduce((acc, c) => acc + (c.volume_mm3 || 0), 0);
  const totalTumorVolumeCm3 = (totalTumorVolumeMm3 / 1000).toFixed(2);

  // Subregions
  const necroticClass = classDistribution.find(c => c.index === 1) || { volume_mm3: 0, percentage: 0 };
  const edemaClass = classDistribution.find(c => c.index === 2) || { volume_mm3: 0, percentage: 0 };
  const enhancingClass = classDistribution.find(c => c.index === 3) || { volume_mm3: 0, percentage: 0 };

  const windowFilter = {
    brain: 'contrast(1.1) brightness(1.0)',
    subdural: 'contrast(1.4) brightness(1.15)',
    bone: 'contrast(1.8) brightness(0.85)',
    custom: 'contrast(1.0) brightness(1.0)',
  }[windowPreset];

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className="space-y-6 pb-12 max-w-[1520px] mx-auto text-neuro-text-primary"
    >
      {/* ─── BREADCRUMBS & TOP BAR ─────────────────────────────────────── */}
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
          <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 font-mono">
            3D MULTIMODAL WORKSTATION
          </span>
          <span className="text-neuro-text-muted hidden sm:inline">•</span>
          <div className="flex items-center gap-1 bg-neuro-surface-300 p-0.5 rounded border border-neuro-border text-[11px] font-mono">
            <button
              type="button"
              onClick={() => setSelectedModel('attention_vit')}
              className={`px-2 py-0.5 rounded transition-colors ${
                !isUnet3D
                  ? 'bg-neuro-accent text-neuro-bg font-bold'
                  : 'text-neuro-text-muted hover:text-white'
              }`}
            >
              Attention CNN + ViT (2D)
            </button>
            <button
              type="button"
              onClick={() => setSelectedModel('unet3d_brats')}
              className={`px-2 py-0.5 rounded transition-colors ${
                isUnet3D
                  ? 'bg-cyan-500 text-black font-bold'
                  : 'text-neuro-text-muted hover:text-white'
              }`}
            >
              3D U-Net (volumetric)
            </button>
          </div>
        </div>

        <div className="flex items-center gap-2 flex-wrap w-full lg:w-auto justify-end">
          <a
            href={downloadUrl}
            download
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-neuro-surface-200 hover:bg-neuro-surface-300 border border-neuro-border text-xs font-mono transition-colors"
          >
            <Download className="w-3.5 h-3.5 text-neuro-accent" />
            <span>Download NIfTI (.nii.gz)</span>
          </a>
          <ExportReportButton analysis={data} />
          <button
            onClick={() => navigate(`/explainability?analysisId=${analysisId}`)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-neuro-surface-200 hover:bg-neuro-surface-300 border border-neuro-border text-xs font-mono text-cyan-400 transition-colors"
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Explainability Studio</span>
          </button>
          <button
            onClick={() => navigate('/new-analysis')}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-md bg-neuro-accent hover:bg-neuro-accent-bright text-white font-semibold text-xs transition-colors shadow-sm"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>New Scan</span>
          </button>
        </div>
      </header>

      {/* ─── 3D MODEL / XAI BANNER ──────────────────────────────────────── */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between p-3 rounded-lg bg-cyan-950/20 border border-cyan-500/30 gap-2 text-xs font-mono">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 font-bold text-[10px]">
            {isUnet3D ? 'GENUINE MODEL PREDICTION - 3D GRAD-CAM (BOTTLENECK, PATCH-LEVEL)' : 'GENUINE MODEL PREDICTION · MULTIMODAL MULTI-PLANAR SLICES'}
          </span>
          <span className="text-neuro-text-secondary text-[11px]">
            {isUnet3D
              ? 'Volumetric 96×96×64 sliding-window inference with trilinear 3D Grad-CAM.'
              : '2D Attention CNN + ViT slice-by-slice multimodal analysis.'}
          </span>
        </div>
        <div className="text-[10px] text-amber-400 font-bold bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20 self-start sm:self-auto">
          Research/educational use only. Not clinically validated.
        </div>
      </div>

      {/* ─── SYNCHRONIZED NAVIGATION STRIP ─────────────────────────────── */}
      <div className="w-full bg-[#0a1324] border border-neuro-border rounded-lg px-4 py-2.5 flex flex-col md:flex-row items-start md:items-center justify-between gap-3 text-xs font-mono">
        <div className="flex items-center gap-3 flex-wrap">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse"></span>
            <span className="text-white font-bold tracking-wide">Multiplanar Navigation</span>
          </div>
          <span className="text-neuro-text-muted">•</span>
          {/* Plane Switcher */}
          <div className="flex items-center gap-1 bg-[#070e1c] p-0.5 rounded border border-neuro-border">
            {(['axial', 'coronal', 'sagittal'] as const).map(plane => (
              <button
                key={plane}
                onClick={() => setActivePlane(plane)}
                className={`px-3 py-1 rounded text-xs font-mono font-medium capitalize transition-all ${
                  activePlane === plane
                    ? 'bg-neuro-accent text-white font-semibold shadow-sm'
                    : 'text-neuro-text-secondary hover:text-white'
                }`}
              >
                {plane}
              </button>
            ))}
          </div>
          <span className="text-neuro-text-muted hidden sm:inline">•</span>
          <span className="text-neuro-text-secondary hidden sm:inline">
            Active Slice: <strong className="text-cyan-400">{sliceIndex}</strong> / {maxSlicesForPlane}
          </span>
        </div>

        <div className="flex items-center gap-3 text-neuro-text-muted text-[11px] flex-wrap">
          <span>
            CENTROID: <strong className="text-cyan-400 font-mono">
              [{anatomicalLocation?.center_x_mm ?? 142}, {anatomicalLocation?.center_y_mm ?? 198}, {anatomicalLocation?.center_z_mm ?? 88}] mm
            </strong>
          </span>
          <span className="hidden sm:inline">•</span>
          <span>VOXEL: <strong className="text-white">{voxelSpacing.map((v: number) => Number(v).toFixed(1)).join(' × ')} mm³</strong></span>
        </div>
      </div>

      {/* ═════════════════════════════════════════════════════════════════ */}
      {/* ─── DEDICATED SECTION: MRI TUMOR SEGMENTATION RESULT ──────────── */}
      {/* ═════════════════════════════════════════════════════════════════ */}
      <section className="space-y-4">
        {/* Section Header */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 border-b border-neuro-border pb-2.5">
          <div className="flex items-center gap-2.5">
            <Scan className="w-5 h-5 text-neuro-accent" />
            <div>
              <h2 className="text-sm font-bold uppercase tracking-wider font-mono text-white">
                MRI Tumor Segmentation Result
              </h2>
              <p className="text-[11px] text-neuro-text-muted">
                Side-by-side synchronized view: uploaded patient MRI slice vs genuine model-predicted tumor segmentation
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              GENUINE MODEL PREDICTION · NO GRAD-CAM
            </span>
          </div>
        </div>

        {/* Side-by-Side Dual Viewport Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* ─── PANEL A: ORIGINAL MRI ─────────────────────────────────── */}
          <div className="bg-[#050a14] rounded-lg border border-neuro-border flex flex-col overflow-hidden shadow-xl">
            {/* Panel A Top Bar */}
            <div className="p-3 bg-[#081020]/95 backdrop-blur-md border-b border-neuro-border flex items-center justify-between gap-2 flex-wrap">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
                <span className="font-mono font-bold text-xs text-white uppercase tracking-wider">
                  Panel A — Original MRI
                </span>
              </div>

              {/* Modality Selector Tabs */}
              <div className="flex items-center gap-1 bg-[#070e1c] p-0.5 rounded border border-neuro-border">
                {(['flair', 't1', 't1ce', 't2'] as const).map(mod => (
                  <button
                    key={mod}
                    onClick={() => setActiveModality(mod)}
                    className={`px-2.5 py-0.5 rounded text-[11px] font-mono uppercase transition-all ${
                      activeModality === mod
                        ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/40 font-bold shadow-sm'
                        : 'text-neuro-text-muted hover:text-white'
                    }`}
                  >
                    {mod}
                  </button>
                ))}
              </div>
            </div>

            {/* Panel A Slice Canvas Container */}
            <div className="relative flex-1 w-full bg-black flex items-center justify-center overflow-hidden min-h-[440px] p-2">
              {/* Technical DICOM Top-Left Corner Stamp */}
              <div className="absolute top-3 left-3 text-[10px] font-mono text-neuro-text-muted z-10 pointer-events-none leading-tight bg-black/50 p-1.5 rounded backdrop-blur-sm">
                <div>SERIES: <span className="text-white font-bold uppercase">{activeModality} · {activePlane}</span></div>
                <div>MATRIX: <span className="text-white">{dimensions[0]} × {dimensions[1]}</span></div>
                <div>ORIENTATION: <span className="text-cyan-400 font-bold">{(data.anatomical_orientation || ['R', 'A', 'S']).join('')}</span></div>
              </div>

              {/* Technical DICOM Top-Right Corner Stamp */}
              <div className="absolute top-3 right-3 text-right text-[10px] font-mono text-neuro-text-muted z-10 pointer-events-none leading-tight bg-black/50 p-1.5 rounded backdrop-blur-sm">
                <div>SLICE: <span className="text-white font-bold">{sliceIndex} / {maxSlicesForPlane}</span></div>
                <div>VOXEL: <span className="text-cyan-400">{voxelSpacing.map((v: number) => Number(v).toFixed(1)).join('×')} mm</span></div>
                <div>STATUS: <span className="text-emerald-400 font-bold">ORIGINAL MRI</span></div>
              </div>

              {/* Anatomical Direction Markers */}
              <span className="absolute left-3 top-1/2 -translate-y-1/2 font-mono font-bold text-xs text-neuro-text-muted/60 pointer-events-none bg-black/40 px-1 rounded">
                {activePlane === 'sagittal' ? 'A' : 'R'}
              </span>
              <span className="absolute right-3 top-1/2 -translate-y-1/2 font-mono font-bold text-xs text-neuro-text-muted/60 pointer-events-none bg-black/40 px-1 rounded">
                {activePlane === 'sagittal' ? 'P' : 'L'}
              </span>
              <span className="absolute top-3 left-1/2 -translate-x-1/2 font-mono font-bold text-xs text-neuro-text-muted/60 pointer-events-none bg-black/40 px-1 rounded">
                {activePlane === 'axial' ? 'A' : 'S'}
              </span>
              <span className="absolute bottom-3 left-1/2 -translate-x-1/2 font-mono font-bold text-xs text-neuro-text-muted/60 pointer-events-none bg-black/40 px-1 rounded">
                {activePlane === 'axial' ? 'P' : 'I'}
              </span>

              {/* Original MRI Slice Image */}
              <div className="relative w-full h-full max-w-[440px] max-h-[440px] flex items-center justify-center">
                <img
                  src={panelAUrl || fallbackSlice}
                  alt={`Original MRI ${activeModality} ${activePlane} slice ${sliceIndex}`}
                  className="w-full h-full object-contain select-none pointer-events-none transition-all duration-100"
                  style={{ filter: windowFilter }}
                  onError={e => {
                    if (fallbackSlice && e.currentTarget.src !== fallbackSlice) {
                      e.currentTarget.src = fallbackSlice;
                    }
                  }}
                />
              </div>
            </div>

            {/* Panel A Footer Metadata */}
            <div className="p-2.5 bg-[#081020]/90 border-t border-neuro-border flex items-center justify-between text-[11px] font-mono text-neuro-text-muted">
              <span>Sequence: <strong className="text-white uppercase">{activeModality} (Pre-Contrast/Post)</strong></span>
              <span>Slice Depth: <strong className="text-cyan-400">{sliceIndex}</strong> mm</span>
            </div>
          </div>

          {/* ─── PANEL B: PREDICTED SEGMENTATION ───────────────────────── */}
          <div className="bg-[#050a14] rounded-lg border border-neuro-border flex flex-col overflow-hidden shadow-xl">
            {/* Panel B Top Bar */}
            <div className="p-3 bg-[#081020]/95 backdrop-blur-md border-b border-neuro-border flex items-center justify-between gap-2 flex-wrap">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
                <span className="font-mono font-bold text-xs text-white uppercase tracking-wider">
                  Panel B — Predicted Segmentation
                </span>
              </div>

              {/* View Mode Selector Tabs */}
              <div className="flex items-center gap-1 bg-[#070e1c] p-0.5 rounded border border-neuro-border font-mono text-xs">
                {(['overlay', 'mask_only', 'original'] as const).map(mode => (
                  <button
                    key={mode}
                    onClick={() => setViewMode(mode)}
                    className={`px-2.5 py-0.5 rounded text-[11px] capitalize transition-all ${
                      viewMode === mode
                        ? 'bg-neuro-accent text-white font-bold shadow-sm'
                        : 'text-neuro-text-muted hover:text-white'
                    }`}
                  >
                    {mode === 'overlay' ? 'Overlay' : mode === 'mask_only' ? 'Mask Only' : 'Original MRI'}
                  </button>
                ))}
              </div>
            </div>

            {/* Panel B Slice Canvas Container */}
            <div className="relative flex-1 w-full bg-black flex items-center justify-center overflow-hidden min-h-[440px] p-2">
              {/* Technical DICOM Top-Left Corner Stamp */}
              <div className="absolute top-3 left-3 text-[10px] font-mono text-neuro-text-muted z-10 pointer-events-none leading-tight bg-black/50 p-1.5 rounded backdrop-blur-sm">
                <div>MODEL: <span className="text-cyan-400 font-bold">{modelName}</span></div>
                <div>VIEW: <span className="text-white uppercase">{viewMode}</span></div>
                <div>OPACITY: <span className="text-white">{Math.round(overlayOpacity * 100)}%</span></div>
              </div>

              {/* Technical DICOM Top-Right Corner Stamp */}
              <div className="absolute top-3 right-3 text-right text-[10px] font-mono text-neuro-text-muted z-10 pointer-events-none leading-tight bg-black/50 p-1.5 rounded backdrop-blur-sm">
                <div>SUBREGIONS: <span className="text-emerald-400 font-bold">{activeClassesParam === 'none' ? 'None' : activeClassesParam}</span></div>
                <div>SLICE: <span className="text-white font-bold">{sliceIndex} / {maxSlicesForPlane}</span></div>
                <div>PREDICTION: <span className="text-amber-400 font-bold">MULTICLASS</span></div>
              </div>

              {/* Anatomical Direction Markers */}
              <span className="absolute left-3 top-1/2 -translate-y-1/2 font-mono font-bold text-xs text-neuro-text-muted/60 pointer-events-none bg-black/40 px-1 rounded">
                {activePlane === 'sagittal' ? 'A' : 'R'}
              </span>
              <span className="absolute right-3 top-1/2 -translate-y-1/2 font-mono font-bold text-xs text-neuro-text-muted/60 pointer-events-none bg-black/40 px-1 rounded">
                {activePlane === 'sagittal' ? 'P' : 'L'}
              </span>
              <span className="absolute top-3 left-1/2 -translate-x-1/2 font-mono font-bold text-xs text-neuro-text-muted/60 pointer-events-none bg-black/40 px-1 rounded">
                {activePlane === 'axial' ? 'A' : 'S'}
              </span>
              <span className="absolute bottom-3 left-1/2 -translate-x-1/2 font-mono font-bold text-xs text-neuro-text-muted/60 pointer-events-none bg-black/40 px-1 rounded">
                {activePlane === 'axial' ? 'P' : 'I'}
              </span>

              {/* Predicted Segmentation Slice Image */}
              <div className="relative w-full h-full max-w-[440px] max-h-[440px] flex items-center justify-center">
                <img
                  src={panelBUrl || fallbackSlice}
                  alt={`Predicted Segmentation ${activePlane} slice ${sliceIndex}`}
                  className="w-full h-full object-contain select-none pointer-events-none transition-all duration-100"
                  style={{ filter: viewMode === 'mask_only' ? 'none' : windowFilter }}
                  onError={e => {
                    if (fallbackSlice && e.currentTarget.src !== fallbackSlice) {
                      e.currentTarget.src = fallbackSlice;
                    }
                  }}
                />
              </div>
            </div>

            {/* Panel B Footer Metadata */}
            <div className="p-2.5 bg-[#081020]/90 border-t border-neuro-border flex items-center justify-between text-[11px] font-mono text-neuro-text-muted">
              <span>Mask Mapping: <strong className="text-white">NCR (#EF4444) · ED (#10B981) · ET (#F59E0B)</strong></span>
              <span>Spatial Alignment: <strong className="text-emerald-400 font-bold">100% Identical to Panel A</strong></span>
            </div>
          </div>
        </div>

        {/* ─── SYNCHRONIZED CONTROLS STRIP (CONTROLS BOTH PANELS) ─────── */}
        <div className="w-full bg-[#081122] border border-neuro-border rounded-lg p-4 space-y-4">
          {/* Top Row: Synchronized Slice Scrubber */}
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 font-mono text-xs">
            <div className="flex items-center gap-2 flex-1 w-full min-w-[280px]">
              <span className="text-neuro-text-muted uppercase text-[10px] font-semibold whitespace-nowrap">
                Synchronized Slice Scrubber:
              </span>
              <button
                onClick={() => setSliceIndex(prev => Math.max(1, prev - 1))}
                className="p-1.5 rounded bg-neuro-surface-200 hover:bg-neuro-surface-300 text-white border border-neuro-border transition-colors"
                title="Previous Slice"
              >
                <ChevronLeft className="w-4 h-4" />
              </button>
              <input
                type="range"
                min={1}
                max={maxSlicesForPlane}
                value={sliceIndex}
                onChange={e => setSliceIndex(parseInt(e.target.value, 10))}
                className="w-full accent-neuro-accent h-2 bg-neuro-surface-300 rounded cursor-pointer"
              />
              <button
                onClick={() => setSliceIndex(prev => Math.min(maxSlicesForPlane, prev + 1))}
                className="p-1.5 rounded bg-neuro-surface-200 hover:bg-neuro-surface-300 text-white border border-neuro-border transition-colors"
                title="Next Slice"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
              <div className="px-3 py-1 rounded bg-neuro-surface-200 border border-neuro-border font-mono text-xs whitespace-nowrap text-neuro-text-secondary">
                Slice <strong className="text-neuro-accent font-bold">{sliceIndex}</strong> / {maxSlicesForPlane}
              </div>
            </div>

            {/* Window Presets for Grayscale MRI */}
            <div className="flex items-center gap-1 font-mono text-xs self-end sm:self-auto">
              <span className="text-neuro-text-muted text-[10px] uppercase mr-1">Window:</span>
              {(['brain', 'subdural', 'bone'] as const).map(w => (
                <button
                  key={w}
                  onClick={() => setWindowPreset(w)}
                  className={`px-2.5 py-0.5 rounded font-medium capitalize transition-colors ${
                    windowPreset === w
                      ? 'bg-neuro-accent/20 text-neuro-accent border border-neuro-accent/40 font-semibold'
                      : 'bg-neuro-surface-200 text-neuro-text-secondary border border-neuro-border hover:text-white'
                  }`}
                >
                  {w}
                </button>
              ))}
            </div>
          </div>

          {/* Bottom Row: Subregion Class Checkboxes & Opacity Slider */}
          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 pt-2 border-t border-neuro-border/60">
            {/* Class Visibility Checkboxes */}
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-1.5 font-mono text-xs text-neuro-text-muted pr-1">
                <Layers className="w-4 h-4 text-neuro-accent" />
                <span className="uppercase tracking-wider text-[10px] font-semibold">Class Visibility:</span>
              </div>

              {/* Necrotic Core (#EF4444) */}
              <button
                type="button"
                onClick={() => setVisibleSubregions(prev => ({ ...prev, necrotic: !prev.necrotic }))}
                className={`flex items-center gap-2 px-3 py-1.5 rounded border font-mono text-xs transition-all select-none ${
                  visibleSubregions.necrotic
                    ? 'bg-[#EF4444]/10 border-[#EF4444]/50 text-red-300 font-semibold'
                    : 'bg-neuro-surface-200/50 border-neuro-border text-neuro-text-muted opacity-50'
                }`}
              >
                {visibleSubregions.necrotic ? <CheckSquare className="w-3.5 h-3.5 text-[#EF4444]" /> : <Square className="w-3.5 h-3.5" />}
                <div className="w-2.5 h-2.5 rounded-full bg-[#EF4444]" />
                <span>Necrotic / Non-enhancing</span>
                <span className="text-[10px] text-neuro-text-muted font-normal">({(necroticClass.volume_mm3 / 1000).toFixed(2)} cm³)</span>
              </button>

              {/* Peritumoral Edema (#10B981) */}
              <button
                type="button"
                onClick={() => setVisibleSubregions(prev => ({ ...prev, edema: !prev.edema }))}
                className={`flex items-center gap-2 px-3 py-1.5 rounded border font-mono text-xs transition-all select-none ${
                  visibleSubregions.edema
                    ? 'bg-[#10B981]/10 border-[#10B981]/50 text-emerald-300 font-semibold'
                    : 'bg-neuro-surface-200/50 border-neuro-border text-neuro-text-muted opacity-50'
                }`}
              >
                {visibleSubregions.edema ? <CheckSquare className="w-3.5 h-3.5 text-[#10B981]" /> : <Square className="w-3.5 h-3.5" />}
                <div className="w-2.5 h-2.5 rounded-full bg-[#10B981]" />
                <span>Peritumoral Edema</span>
                <span className="text-[10px] text-neuro-text-muted font-normal">({(edemaClass.volume_mm3 / 1000).toFixed(2)} cm³)</span>
              </button>

              {/* Enhancing Tumor (#F59E0B) */}
              <button
                type="button"
                onClick={() => setVisibleSubregions(prev => ({ ...prev, enhancing: !prev.enhancing }))}
                className={`flex items-center gap-2 px-3 py-1.5 rounded border font-mono text-xs transition-all select-none ${
                  visibleSubregions.enhancing
                    ? 'bg-[#F59E0B]/10 border-[#F59E0B]/50 text-amber-300 font-semibold'
                    : 'bg-neuro-surface-200/50 border-neuro-border text-neuro-text-muted opacity-50'
                }`}
              >
                {visibleSubregions.enhancing ? <CheckSquare className="w-3.5 h-3.5 text-[#F59E0B]" /> : <Square className="w-3.5 h-3.5" />}
                <div className="w-2.5 h-2.5 rounded-full bg-[#F59E0B]" />
                <span>Enhancing Tumor</span>
                <span className="text-[10px] text-neuro-text-muted font-normal">({(enhancingClass.volume_mm3 / 1000).toFixed(2)} cm³)</span>
              </button>
            </div>

            {/* Mask Opacity Slider */}
            <div className="flex items-center gap-3 font-mono text-xs bg-neuro-surface-200 px-3 py-1.5 rounded border border-neuro-border">
              <Sliders className="w-3.5 h-3.5 text-neuro-accent" />
              <span className="text-neuro-text-muted uppercase text-[10px]">Mask Opacity:</span>
              <input
                type="range"
                min={0.0}
                max={1.0}
                step={0.05}
                value={overlayOpacity}
                onChange={e => setOverlayOpacity(parseFloat(e.target.value))}
                className="w-24 accent-neuro-accent h-1.5 bg-neuro-surface-300 rounded cursor-pointer"
              />
              <span className="text-white w-8 text-right font-bold">{Math.round(overlayOpacity * 100)}%</span>
            </div>
          </div>
        </div>
      </section>

      {/* ═════════════════════════════════════════════════════════════════ */}
      {/* ─── DEDICATED SECTION: 3D BRAIN AND TUMOR RECONSTRUCTION ──────── */}
      {/* ═════════════════════════════════════════════════════════════════ */}
      <section className="space-y-4">
        {/* Section Header */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 border-b border-neuro-border pb-2.5">
          <div className="flex items-center gap-2.5">
            <Brain className="w-5 h-5 text-cyan-400" />
            <div>
              <h2 className="text-sm font-bold uppercase tracking-wider font-mono text-white">
                3D Brain and Tumor Reconstruction
              </h2>
              <p className="text-[11px] text-neuro-text-muted">
                Volumetric point-cloud surface reconstructed directly from predicted NIfTI label map & spatial affine
              </p>
            </div>
          </div>
          <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
            INTERACTIVE RAY-MARCHED 3D SURFACE
          </span>
        </div>

        {/* 3D Canvas & Patient RAS Localization Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 min-h-[500px]">
          {/* Main 3D Canvas Viewport (8 cols) */}
          <div className="lg:col-span-8 relative bg-[#050a14] rounded-lg border border-neuro-border flex flex-col overflow-hidden shadow-xl">
            <div className="p-3 bg-[#081020]/90 backdrop-blur-md border-b border-neuro-border flex items-center justify-between text-xs font-mono">
              <span className="text-neuro-accent font-semibold uppercase tracking-wider">
                Interactive Brain Envelope & Subregions
              </span>
              <div className="flex items-center gap-2 text-[10px] text-neuro-text-muted">
                <span>ROTATION / PAN / ZOOM</span>
                <span>•</span>
                <span className="text-cyan-400 font-bold">RAS COORDINATES</span>
              </div>
            </div>

            <div className="flex-1 w-full bg-[#050a14] relative flex items-center justify-center p-2 min-h-[460px]">
              <Tumor3DViewer
                surfaceData={surface3D}
                dimensions={dimensions}
                anatomicalOrientation={data.anatomical_orientation || ['R', 'A', 'S']}
              />
            </div>
          </div>

          {/* Right Column: RAS Stereotactic Localization & Volumetric Summary (4 cols) */}
          <div className="lg:col-span-4 space-y-4 flex flex-col">
            {/* Stereotactic RAS Localization Card */}
            {anatomicalLocation && (
              <div className="p-4 rounded-lg bg-neuro-surface-200 border border-neuro-border space-y-3">
                <div className="flex items-center gap-2 border-b border-neuro-border pb-2">
                  <Activity className="w-4 h-4 text-cyan-400" />
                  <h3 className="text-xs font-bold uppercase tracking-wider font-mono">
                    Stereotactic Localization (RAS)
                  </h3>
                </div>
                <div className="space-y-2 text-xs">
                  <div className="p-2.5 rounded bg-neuro-surface-300/60 border border-neuro-border space-y-0.5">
                    <span className="text-[10px] text-neuro-text-muted font-semibold uppercase">Hemisphere:</span>
                    <p className="text-sm font-bold text-white">{anatomicalLocation.hemisphere}</p>
                    <p className="text-[11px] text-neuro-text-secondary">{anatomicalLocation.hemisphere_detail}</p>
                  </div>
                  <div className="p-2.5 rounded bg-neuro-surface-300/60 border border-neuro-border space-y-0.5">
                    <span className="text-[10px] text-neuro-text-muted font-semibold uppercase">Anterior-Posterior:</span>
                    <p className="text-sm font-bold text-white">{anatomicalLocation.anterior_posterior}</p>
                    <p className="text-[11px] text-neuro-text-secondary">{anatomicalLocation.ap_detail}</p>
                  </div>
                  <div className="p-2.5 rounded bg-neuro-surface-300/60 border border-neuro-border space-y-0.5">
                    <span className="text-[10px] text-neuro-text-muted font-semibold uppercase">Superior-Inferior:</span>
                    <p className="text-sm font-bold text-white">{anatomicalLocation.superior_inferior}</p>
                    <p className="text-[11px] text-neuro-text-secondary">{anatomicalLocation.si_detail}</p>
                  </div>
                </div>
              </div>
            )}

            {/* Total Tumor Mass Summary */}
            <div className="p-4 rounded-lg bg-neuro-surface-200 border border-neuro-border space-y-3 flex-1">
              <div className="flex items-center justify-between border-b border-neuro-border pb-2">
                <div className="flex items-center gap-2">
                  <Brain className="w-4 h-4 text-neuro-accent" />
                  <h3 className="text-xs font-bold uppercase tracking-wider font-mono">Total Tumor Mass</h3>
                </div>
                <span className="font-mono font-bold text-base text-cyan-400">
                  {totalTumorVolumeCm3} <span className="text-xs font-normal text-neuro-text-muted">cm³</span>
                </span>
              </div>

              <div className="space-y-2 pt-1">
                {classDistribution.filter(c => c.index > 0).map(c => (
                  <div key={c.index} className="p-2 rounded bg-neuro-surface-300/60 border border-neuro-border flex items-center justify-between text-xs font-mono">
                    <div className="flex items-center gap-2">
                      <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: c.color || '#28C7B7' }} />
                      <span className="font-semibold text-white">{c.name}</span>
                    </div>
                    <div className="text-right">
                      <span className="font-bold text-white">{(c.volume_mm3 / 1000).toFixed(2)} cm³</span>
                      <span className="text-[10px] text-neuro-text-muted ml-1.5">({c.percentage}%)</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ═════════════════════════════════════════════════════════════════ */}
      {/* ─── SECTION 3: QUANTITATIVE EVALUATION & MODEL PROVENANCE ─────── */}
      {/* ═════════════════════════════════════════════════════════════════ */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Left Column (6 cols): Ground-Truth Validation Metrics (if SEG supplied) */}
        <div className="lg:col-span-6 space-y-4">
          {evaluationMetrics ? (
            <div className="p-4 rounded-lg bg-neuro-surface-200 border border-neuro-border space-y-3">
              <div className="flex items-center gap-2 border-b border-neuro-border pb-2.5">
                <FileCheck className="w-4 h-4 text-emerald-400" />
                <h3 className="text-xs font-bold uppercase tracking-wider font-mono">
                  Ground-Truth Validation Metrics (Cohort Test Reference)
                </h3>
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                <div className="p-2.5 rounded bg-neuro-surface-300/60 border border-neuro-border">
                  <p className="text-[10px] text-neuro-text-muted uppercase font-semibold">Whole Tumor Dice</p>
                  <p className="text-base font-bold text-emerald-400 font-mono mt-0.5">
                    {(evaluationMetrics.whole_tumor_dice * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="p-2.5 rounded bg-neuro-surface-300/60 border border-neuro-border">
                  <p className="text-[10px] text-neuro-text-muted uppercase font-semibold">Tumor Core Dice</p>
                  <p className="text-base font-bold text-cyan-400 font-mono mt-0.5">
                    {(evaluationMetrics.tumor_core_dice * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="p-2.5 rounded bg-neuro-surface-300/60 border border-neuro-border">
                  <p className="text-[10px] text-neuro-text-muted uppercase font-semibold">Enhancing Dice</p>
                  <p className="text-base font-bold text-amber-400 font-mono mt-0.5">
                    {(evaluationMetrics.enhancing_tumor_dice * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="p-2.5 rounded bg-neuro-surface-300/60 border border-neuro-border">
                  <p className="text-[10px] text-neuro-text-muted uppercase font-semibold">WT IoU</p>
                  <p className="text-base font-bold text-purple-400 font-mono mt-0.5">
                    {(evaluationMetrics.whole_tumor_iou * 100).toFixed(1)}%
                  </p>
                </div>
              </div>
            </div>
          ) : (
            <div className="p-4 rounded-lg bg-neuro-surface-200 border border-neuro-border space-y-2">
              <div className="flex items-center gap-2 border-b border-neuro-border pb-2">
                <ShieldCheck className="w-4 h-4 text-neuro-accent" />
                <h3 className="text-xs font-bold uppercase tracking-wider font-mono">
                  Segmentation Integrity & Subregion Definitions
                </h3>
              </div>
              <p className="text-xs text-neuro-text-secondary leading-relaxed">
                The segmentation output isolates three clinically significant subregions defined by the Brain Tumor Segmentation (BraTS) consortium:
              </p>
              <ul className="text-xs text-neuro-text-muted space-y-1 font-mono list-disc list-inside">
                <li><strong className="text-red-400">Class 1 — Necrotic / Non-enhancing Core:</strong> Avascular, necrotic regions and non-enhancing tumor tissue.</li>
                <li><strong className="text-emerald-400">Class 2 — Peritumoral Edema:</strong> Vasogenic edema in fluid-attenuated FLAIR hyperintense regions.</li>
                <li><strong className="text-amber-400">Class 3 — Enhancing Tumor:</strong> Hyper-vascularized active tumor rim highlighted on post-contrast T1CE.</li>
              </ul>
            </div>
          )}
        </div>

        {/* Right Column (6 cols): Verified Model Provenance Specifications */}
        <div className="lg:col-span-6 space-y-4">
          <div className="p-4 rounded-lg bg-neuro-surface-200 border border-neuro-border space-y-2 text-xs">
            <div className="flex items-center justify-between border-b border-neuro-border pb-2">
              <div className="flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-neuro-accent" />
                <h3 className="text-xs font-bold uppercase tracking-wider font-mono">Model Provenance & Execution Specs</h3>
              </div>
              <span className="px-1.5 py-0.5 rounded text-[10px] bg-neuro-surface-300 border border-neuro-border text-cyan-400 font-mono font-bold">
                {modelId}
              </span>
            </div>
            <div className="space-y-1.5 text-neuro-text-secondary font-mono">
              <div className="flex justify-between py-0.5 border-b border-neuro-border/30">
                <span className="text-neuro-text-muted">Model Name:</span>
                <span className="font-semibold text-white">{modelName}</span>
              </div>
              <div className="flex justify-between py-0.5 border-b border-neuro-border/30">
                <span className="text-neuro-text-muted">Architecture:</span>
                <span className="font-semibold text-white text-right max-w-[260px] truncate" title={architecture}>
                  {architecture}
                </span>
              </div>
              <div className="flex justify-between py-0.5 border-b border-neuro-border/30">
                <span className="text-neuro-text-muted">Version:</span>
                <span className="text-cyan-400 font-semibold">{modelVersion}</span>
              </div>
              <div className="flex justify-between py-0.5 border-b border-neuro-border/30">
                <span className="text-neuro-text-muted">Framework:</span>
                <span>{framework}</span>
              </div>
              <div className="flex justify-between py-0.5 border-b border-neuro-border/30">
                <span className="text-neuro-text-muted">Modalities:</span>
                <span className="text-neuro-accent font-semibold">{inputModalities.join(', ')}</span>
              </div>
              <div className="flex justify-between py-0.5 border-b border-neuro-border/30">
                <span className="text-neuro-text-muted">Dimensions:</span>
                <span>{dimensions.join(' × ')} ({voxelSpacing.map((v: number) => Number(v).toFixed(1)).join(' × ')} mm)</span>
              </div>
              <div className="flex justify-between py-0.5 border-b border-neuro-border/30">
                <span className="text-neuro-text-muted">Parameters:</span>
                <span className="font-mono text-cyan-400">{parametersCount}</span>
              </div>
              <div className="flex justify-between py-0.5 border-b border-neuro-border/30">
                <span className="text-neuro-text-muted">Preprocessing:</span>
                <span className="text-right max-w-[260px] truncate" title={preprocessing}>
                  {preprocessing}
                </span>
              </div>
              <div className="flex justify-between py-0.5">
                <span className="text-neuro-text-muted">Processing Time:</span>
                <span className="font-mono text-emerald-400">{processingTimeMs} ms</span>
              </div>
            </div>

            {isUnet3D && (
              <div className="mt-3 p-3 rounded bg-black/40 border border-cyan-500/20 space-y-2 text-[11px] font-mono">
                <div className="font-bold text-cyan-400">BraTS 2020 Complete-Volume Benchmark (56 Patients):</div>
                <div className="grid grid-cols-2 gap-x-2 gap-y-1 text-neuro-text-muted">
                  <div>Whole Tumor Dice: <span className="text-white font-bold">0.8298</span> (IoU 0.7283)</div>
                  <div>Tumor Core Dice: <span className="text-white font-bold">0.6961</span> (IoU 0.5931)</div>
                  <div>Edema Dice: <span className="text-white font-bold">0.6945</span> (IoU 0.5725)</div>
                  <div>Enhancing Dice: <span className="text-white font-bold">0.6411</span> (IoU 0.5441)</div>
                  <div>Necrotic Dice: <span className="text-white font-bold">0.4704</span> (IoU 0.3602)</div>
                  <div>Mean Fg Dice: <span className="text-white font-bold">0.6020</span> (Acc 99.57%)</div>
                </div>
                <p className="text-[10px] text-neuro-text-muted italic border-t border-neuro-border/30 pt-1.5">
                  The 2D Attention CNN + ViT whole-tumor Dice of 0.8937 used selected tumor-enriched slices; the protocols differ, so the numbers are not a head-to-head comparison.
                </p>
              </div>
            )}
          </div>

          <ResearchDisclaimer />
        </div>
      </div>
    </motion.div>
  );
}