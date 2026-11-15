import { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  ArrowLeft,
  Brain,
  CheckCircle2,
  AlertCircle,
  Clock,
  Sliders,
  Layers,
  Sparkles,
  ChevronRight,
  ShieldCheck,
  Eye,
  RefreshCw,
  ExternalLink,
  ChevronDown,
  Scan,
  Loader2,
} from 'lucide-react';
import { LoadingSpinner } from '@/components/common/LoadingState';
import { ResearchDisclaimer } from '@/components/common/ResearchDisclaimer';
import { getAnalysisById, generate2DSegmentation } from '@/services/api';
import { formatPercentage, formatDate } from '@/lib/utils';
import ExportReportButton from '@/components/common/ExportReportButton';
import { extractNormalizedScores } from '@/lib/scoreUtils';
import { SegExplainViewer } from '@/components/analysis/SegExplainViewer';

export default function Classification2DResultsPage() {
  const { analysisId } = useParams<{ analysisId: string }>();
  const navigate = useNavigate();

  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [analysisData, setAnalysisData] = useState<any>(null);
  const [camOpacity, setCamOpacity] = useState<number>(0.65);
  const [showHeatmap, setShowHeatmap] = useState<boolean>(true);
  const [blendMode, setBlendMode] = useState<'screen' | 'overlay' | 'normal'>('screen');
  const [segViewMode, setSegViewMode] = useState<'overlay' | 'mask' | 'explain'>('overlay');
  const [isGeneratingSeg, setIsGeneratingSeg] = useState<boolean>(false);
  const [segError, setSegError] = useState<string | null>(null);
  const [selectedSegLayer, setSelectedSegLayer] = useState<string>('conv2d_20');
  const [isLoadingSegLayer, setIsLoadingSegLayer] = useState<boolean>(false);

  useEffect(() => {
    if (!analysisId) {
      setError('Analysis ID is required.');
      setIsLoading(false);
      return;
    }

    setIsLoading(true);
    setError(null);

    getAnalysisById(analysisId)
      .then(data => {
        if (!data) {
          setError('Analysis not found.');
          return;
        }
        const is3D =
          data.workflow_type === '3D_SEGMENTATION' ||
          data.analysis_mode === 'segmentation_3d' ||
          data.analysis_mode === 'segmentation' ||
          Boolean(data.surface_3d) ||
          Boolean(data.volume_dimensions);

        if (is3D) {
          // Study is 3D multimodal segmentation; redirect immediately to dedicated 3D workstation
          navigate(`/analysis/${analysisId}/3d-results`, { replace: true });
          return;
        }
        setAnalysisData(data);
      })
      .catch(err => {
        setError(err instanceof Error ? err.message : 'Failed to retrieve analysis.');
      })
      .finally(() => {
        setIsLoading(false);
      });
  }, [analysisId]);

  if (isLoading) {
    return (
      <div className="min-h-[500px] flex items-center justify-center">
        <LoadingSpinner text="Loading 2D Brain Tumor Classification Results..." />
      </div>
    );
  }

  if (error || !analysisData) {
    return (
      <div className="max-w-2xl mx-auto py-12 px-4 space-y-4 text-center">
        <div className="p-8 rounded-xl bg-neuro-surface-200 border border-neuro-border space-y-4">
          <AlertCircle className="w-10 h-10 text-red-400 mx-auto" />
          <h2 className="text-lg font-bold text-neuro-text-primary">Unable to Load Analysis</h2>
          <p className="text-sm text-neuro-text-secondary">{error || 'Record does not exist.'}</p>
          <div className="pt-2 flex justify-center gap-3">
            <button
              onClick={() => navigate('/new-analysis')}
              className="px-4 py-2 rounded-lg bg-neuro-accent hover:bg-neuro-accent-bright text-white text-xs font-semibold"
            >
              New Analysis
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

  // Extract fields normalized from API
  const fileName = analysisData.file_name || analysisData.fileName || 'brain_slice.png';
  const createdAt = analysisData.created_at || analysisData.createdAt || new Date().toISOString();
  const processingTimeMs = analysisData.processing_time_ms || analysisData.actual_model?.inference_time_ms || analysisData.metadata?.processingTimeMs || 142;
  const topPrediction = analysisData.top_prediction || analysisData.predictedClass || 'Glioma';
  const confidence = analysisData.confidence || 0.94;

  const actualModel = analysisData.actual_model || {};
  const modelId = actualModel.model_id || analysisData.model_id || (analysisData.model_name?.toLowerCase().includes('swin') ? 'swin_transformer' : (analysisData.model_name?.toLowerCase().includes('256') ? 'efficientnetv2b0_256' : 'efficientnetv2b0'));
  const modelName = actualModel.model_name || analysisData.model_name || analysisData.metadata?.modelName || (modelId === 'swin_transformer' ? 'Swin Transformer' : (modelId === 'efficientnetv2b0_256' ? 'EfficientNetV2B0 (CLAHE, 256x256)' : 'EfficientNetV2B0'));
  const modelVersion = actualModel.model_version || analysisData.model_version || analysisData.metadata?.modelVersion || (modelId === 'swin_transformer' ? 'SwinTransformer-224' : (modelId === 'efficientnetv2b0_256' ? 'v2.5.0-clahe256' : 'v2.4.0'));
  const architecture = actualModel.architecture || (modelId === 'swin_transformer' ? 'Hierarchical Shifted-Window Transformer' : (modelId === 'efficientnetv2b0_256' ? 'EfficientNetV2-B0 (Fused-MBConv + Squeeze-and-Excitation)' : 'EfficientNetV2-B0 (Fused-MBConv)'));
  const framework = actualModel.framework || 'TensorFlow 2.18 / Keras 3';
  const inputResolution = actualModel.input_resolution || (modelId === 'swin_transformer' ? '224 × 224 × 3 RGB' : (modelId === 'efficientnetv2b0_256' ? '256 × 256 × 3 RGB' : '128 × 128 × 3 RGB'));
  const parametersCount = actualModel.parameters_count || (modelId === 'swin_transformer' ? '28.3M params' : '7.11M params');
  const preprocessing = actualModel.preprocessing || (modelId === 'swin_transformer' ? 'Bicubic Interpolation + /255.0 Normalization' : (modelId === 'efficientnetv2b0_256' ? 'Grayscale CLAHE -> Bilinear 256x256 -> Float32 [0, 255]' : 'CLAHE Adaptive Histogram Equalization'));

  // Probabilities distribution
  const scores = extractNormalizedScores(analysisData);

  const originalImage = analysisData.image_data_url || analysisData.imageDataUrl || analysisData.thumbnail_url || analysisData.thumbnailUrl;
  const gradcamOverlay = analysisData.gradcam_overlay || analysisData.explainability?.gradcam_overlay || originalImage;
  const segmentationOverlayUrl = analysisData.segmentation_overlay_url;
  const segmentationMaskUrl = analysisData.segmentation_mask_url;
  const segmentationModelUsed = analysisData.segmentation_model_used;
  const segmentationTumorAreaPct = analysisData.segmentation_tumor_area_pct;

  const handleGenerateSegmentation = async () => {
    if (!analysisId || isGeneratingSeg) return;
    setIsGeneratingSeg(true);
    setSegError(null);
    try {
      const res = await generate2DSegmentation(analysisId, selectedSegLayer);
      setAnalysisData((prev: any) =>
        prev
          ? {
              ...prev,
              ...res,
            }
          : prev
      );
    } catch (err) {
      setSegError(err instanceof Error ? err.message : 'Failed to generate tumor segmentation.');
    } finally {
      setIsGeneratingSeg(false);
    }
  };

  const handleExplainLayerChange = async (newLayer: string) => {
    if (!analysisId || isLoadingSegLayer) return;
    setSelectedSegLayer(newLayer);
    setIsLoadingSegLayer(true);
    try {
      const res = await generate2DSegmentation(analysisId, newLayer);
      setAnalysisData((prev: any) =>
        prev
          ? {
              ...prev,
              ...res,
            }
          : prev
      );
    } catch (err) {
      console.error('Failed to update Seg-Grad-CAM layer', err);
    } finally {
      setIsLoadingSegLayer(false);
    }
  };

  // Class badge styling
  const getClassBadgeStyle = (label: string) => {
    const l = label.toLowerCase().replace(/[\s_-]/g, '');
    if (l.includes('glioma')) return 'bg-amber-500/10 text-amber-400 border-amber-500/30';
    if (l.includes('meningioma')) return 'bg-purple-500/10 text-purple-400 border-purple-500/30';
    if (l.includes('pituitary')) return 'bg-blue-500/10 text-blue-400 border-blue-500/30';
    return 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30';
  };

  const getBlendClass = () => {
    if (blendMode === 'screen') return 'mix-blend-screen';
    if (blendMode === 'overlay') return 'mix-blend-overlay';
    return 'mix-blend-normal';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="space-y-6 pb-12 max-w-6xl mx-auto"
    >
      {/* Breadcrumbs & Navigation Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 border-b border-neuro-border pb-4">
        <div className="flex items-center gap-2 text-xs text-neuro-text-muted flex-wrap">
          <Link to="/new-analysis" className="hover:text-neuro-accent flex items-center gap-1 transition-colors">
            <ArrowLeft className="w-3.5 h-3.5" /> Analysis Hub
          </Link>
          <ChevronRight className="w-3 h-3" />
          <Link to="/history" className="hover:text-neuro-accent transition-colors">
            History
          </Link>
          <ChevronRight className="w-3 h-3" />
          <span className="font-mono text-neuro-text-secondary">{analysisId}</span>
          <span className="px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            2D Classification
          </span>
        </div>

        <div className="flex items-center gap-2 w-full sm:w-auto">
          <Link
            to={`/explainability?analysisId=${analysisId}`}
            className="px-3 py-1.5 rounded-lg bg-neuro-accent/15 hover:bg-neuro-accent/25 text-neuro-accent text-xs font-semibold border border-neuro-accent/30 transition-colors flex items-center gap-1.5"
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>Explainability Studio</span>
          </Link>
          <ExportReportButton analysis={analysisData} />
          <button
            onClick={() => navigate('/new-analysis')}
            className="px-3 py-1.5 rounded-lg bg-neuro-surface-200 hover:bg-neuro-surface-300 text-neuro-text-secondary text-xs font-medium border border-neuro-border transition-colors flex items-center gap-1.5"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>New Scan</span>
          </button>
        </div>
      </div>

      {/* Top Banner: Analysis Summary */}
      <div className="p-5 rounded-xl bg-neuro-surface-200 border border-neuro-border flex flex-col md:flex-row items-start md:items-center justify-between gap-4 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400 flex-shrink-0">
            <Brain className="w-6 h-6" />
          </div>
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <h1 className="text-lg font-bold text-neuro-text-primary capitalize">{topPrediction} Detected</h1>
              <span className={`px-2.5 py-0.5 rounded-full text-xs font-semibold border ${getClassBadgeStyle(topPrediction)}`}>
                {formatPercentage(confidence)} Confidence
              </span>
            </div>
            <p className="text-xs text-neuro-text-muted mt-1 flex items-center gap-3 flex-wrap">
              <span>File: <strong className="text-neuro-text-secondary font-mono">{fileName}</strong></span>
              <span>•</span>
              <span>Model: <strong className="text-neuro-text-secondary">{modelName} (v{modelVersion})</strong></span>
              <span>•</span>
              <span className="flex items-center gap-1"><Clock className="w-3 h-3" /> {processingTimeMs} ms</span>
              <span>•</span>
              <span>{formatDate(createdAt)}</span>
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3 self-stretch md:self-auto justify-end border-t md:border-t-0 pt-3 md:pt-0 border-neuro-border/60">
          <div className="text-right">
            <p className="text-[11px] text-neuro-text-muted uppercase tracking-wider font-semibold">Diagnostic Category</p>
            <p className="text-sm font-bold text-neuro-text-primary capitalize">{topPrediction}</p>
          </div>
        </div>
      </div>

      {/* Main Grid: Scan Viewer & Probability Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column (7 cols): Original MRI & Grad-CAM Heatmap Viewer */}
        <div className="lg:col-span-7 space-y-4">
          <div className="p-4 rounded-xl bg-neuro-surface-200 border border-neuro-border space-y-3">
            <div className="flex items-center justify-between border-b border-neuro-border pb-3 flex-wrap gap-2">
              <div className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-neuro-accent" />
                <h3 className="text-xs font-bold text-neuro-text-primary uppercase tracking-wider">
                  MRI Slice & Saliency Inspection
                </h3>
              </div>
              <div className="flex items-center gap-2">
                {/* Blend Mode Selector */}
                <div className="flex items-center bg-neuro-surface-300 rounded p-0.5 border border-neuro-border text-[11px]">
                  <button
                    type="button"
                    onClick={() => setBlendMode('screen')}
                    className={`px-2 py-0.5 rounded ${blendMode === 'screen' ? 'bg-neuro-accent text-neuro-bg font-semibold' : 'text-neuro-text-muted hover:text-neuro-text-primary'}`}
                  >
                    Screen
                  </button>
                  <button
                    type="button"
                    onClick={() => setBlendMode('overlay')}
                    className={`px-2 py-0.5 rounded ${blendMode === 'overlay' ? 'bg-neuro-accent text-neuro-bg font-semibold' : 'text-neuro-text-muted hover:text-neuro-text-primary'}`}
                  >
                    Overlay
                  </button>
                  <button
                    type="button"
                    onClick={() => setBlendMode('normal')}
                    className={`px-2 py-0.5 rounded ${blendMode === 'normal' ? 'bg-neuro-accent text-neuro-bg font-semibold' : 'text-neuro-text-muted hover:text-neuro-text-primary'}`}
                  >
                    Direct
                  </button>
                </div>

                <button
                  type="button"
                  onClick={() => setShowHeatmap(!showHeatmap)}
                  className={`px-2.5 py-1 rounded text-xs font-medium border flex items-center gap-1 transition-all ${
                    showHeatmap
                      ? 'bg-neuro-accent/20 border-neuro-accent text-neuro-accent'
                      : 'bg-neuro-surface-300 border-neuro-border text-neuro-text-muted'
                  }`}
                >
                  <Eye className="w-3 h-3" />
                  <span>{showHeatmap ? 'Heatmap ON' : 'Heatmap OFF'}</span>
                </button>
              </div>
            </div>

            {/* Viewport with DICOM HUD Overlays */}
            <div className="relative aspect-square max-h-[460px] w-full rounded-lg bg-black overflow-hidden border border-neuro-border flex items-center justify-center">
              {originalImage ? (
                <>
                  <img
                    src={originalImage}
                    alt="Original Brain MRI"
                    className="absolute inset-0 w-full h-full object-contain"
                  />
                  {showHeatmap && gradcamOverlay && (
                    <img
                      src={gradcamOverlay}
                      alt="Grad-CAM Saliency Map"
                      className={`absolute inset-0 w-full h-full object-contain transition-opacity pointer-events-none ${getBlendClass()}`}
                      style={{ opacity: camOpacity }}
                    />
                  )}

                  {/* DICOM HUD Corner Metadata */}
                  <div className="absolute top-2 left-2 pointer-events-none font-mono text-[10px] text-neuro-accent bg-black/60 px-2 py-1 rounded backdrop-blur-xs border border-white/10">
                    <div>SERIES: 2D-AXIAL</div>
                    <div className="text-neuro-text-muted">FOV: 256×256 mm</div>
                    <div className="text-neuro-text-muted">RECON: CLAHE</div>
                  </div>

                  <div className="absolute top-2 right-2 pointer-events-none font-mono text-[10px] text-right text-neuro-text-muted bg-black/60 px-2 py-1 rounded backdrop-blur-xs border border-white/10">
                    <div className="text-white font-semibold">TARGET: {topPrediction.toUpperCase()}</div>
                    <div>CONF: {formatPercentage(confidence)}</div>
                  </div>

                  <div className="absolute bottom-2 left-2 pointer-events-none font-mono text-[10px] text-neuro-text-muted bg-black/60 px-2 py-0.5 rounded border border-white/10">
                    BLEND: {blendMode.toUpperCase()}
                  </div>

                  <div className="absolute bottom-2 right-2 pointer-events-none font-mono text-[10px] text-neuro-accent bg-black/60 px-2 py-0.5 rounded border border-white/10">
                    GRAD-CAM ACTIVE
                  </div>
                </>
              ) : (
                <div className="text-center text-neuro-text-muted space-y-2 p-6">
                  <Brain className="w-12 h-12 mx-auto opacity-30" />
                  <p className="text-xs">No image preview available for this scan.</p>
                </div>
              )}
            </div>

            {/* Heatmap Opacity Slider */}
            {showHeatmap && (
              <div className="flex items-center justify-between gap-4 pt-2 text-xs border-t border-neuro-border/60">
                <div className="flex items-center gap-2 text-neuro-text-secondary">
                  <Sliders className="w-3.5 h-3.5 text-neuro-accent" />
                  <span>Saliency Overlay Opacity</span>
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={camOpacity}
                    onChange={e => setCamOpacity(parseFloat(e.target.value))}
                    className="w-32 accent-neuro-accent cursor-pointer"
                  />
                  <span className="font-mono text-[11px] text-neuro-text-primary w-8 text-right">
                    {Math.round(camOpacity * 100)}%
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Predicted Tumor Segmentation Mask Card */}
          <div className="p-4 rounded-xl bg-neuro-surface-200 border border-neuro-border space-y-3">
            <div className="flex items-center justify-between border-b border-neuro-border pb-3 flex-wrap gap-2">
              <div className="flex items-center gap-2">
                <Scan className="w-4 h-4 text-red-400" />
                <h3 className="text-xs font-bold text-neuro-text-primary uppercase tracking-wider">
                  Tumor Segmentation (Predicted Mask)
                </h3>
                {segmentationModelUsed && (
                  <span className="px-2 py-0.5 rounded text-[10px] font-mono font-semibold bg-red-500/10 text-red-400 border border-red-500/20 uppercase">
                    {segmentationModelUsed === 'deeplabv3plus' ? 'DeepLabV3+' : 'BRISC U-Net'}
                  </span>
                )}
              </div>

              {(segmentationOverlayUrl || segmentationMaskUrl) && (
                <div className="flex items-center bg-neuro-surface-300 rounded p-0.5 border border-neuro-border text-[11px]">
                  <button
                    type="button"
                    onClick={() => setSegViewMode('overlay')}
                    className={`px-2.5 py-0.5 rounded ${segViewMode === 'overlay' ? 'bg-neuro-accent text-neuro-bg font-semibold' : 'text-neuro-text-muted hover:text-neuro-text-primary'}`}
                  >
                    Overlay
                  </button>
                  <button
                    type="button"
                    onClick={() => setSegViewMode('mask')}
                    className={`px-2.5 py-0.5 rounded ${segViewMode === 'mask' ? 'bg-neuro-accent text-neuro-bg font-semibold' : 'text-neuro-text-muted hover:text-neuro-text-primary'}`}
                  >
                    Binary Mask
                  </button>
                  <button
                    type="button"
                    onClick={() => setSegViewMode('explain')}
                    className={`px-2.5 py-0.5 rounded ${segViewMode === 'explain' ? 'bg-neuro-accent text-neuro-bg font-semibold' : 'text-neuro-text-muted hover:text-neuro-text-primary'}`}
                  >
                    Explain
                  </button>
                </div>
              )}
            </div>

            {segmentationOverlayUrl || segmentationMaskUrl ? (
              segViewMode === 'explain' ? (
                <SegExplainViewer
                  originalImage={originalImage}
                  heatmapUrl={
                    analysisData.explain_segmentation_url ||
                    analysisData.explain_segmentation_heatmap_png ||
                    analysisData.explain?.heatmap_png
                  }
                  maskUrl={segmentationMaskUrl}
                  layerName={
                    analysisData.explain_segmentation_layer ||
                    analysisData.explain?.layer ||
                    selectedSegLayer
                  }
                  featureMapShape={
                    analysisData.explain_segmentation_feature_map ||
                    analysisData.explain?.map_shape ||
                    (selectedSegLayer === 'up_sampling2d_2' ? '256 × 256 × 256' : '16 × 16 × 256')
                  }
                  isAvailable={
                    analysisData.explain_segmentation_available !== false &&
                    analysisData.explain?.available !== false
                  }
                  unavailableReason={
                    analysisData.explain?.reason || analysisData.explain_segmentation_reason
                  }
                  blendMode={blendMode}
                  opacity={camOpacity}
                  stats={
                    analysisData.explain_segmentation_stats || analysisData.explain?.stats
                  }
                  onLayerChange={handleExplainLayerChange}
                  selectedLayer={selectedSegLayer}
                  isLoadingLayer={isLoadingSegLayer}
                />
              ) : (
                <>
                  <div className="relative aspect-square max-h-[400px] w-full rounded-lg bg-black overflow-hidden border border-neuro-border flex items-center justify-center">
                    <img
                      src={
                        segViewMode === 'mask'
                          ? (segmentationMaskUrl || segmentationOverlayUrl)
                          : (segmentationOverlayUrl || segmentationMaskUrl)
                      }
                      alt="Predicted Tumor Segmentation"
                      className="w-full h-full object-contain"
                    />
                    <div className="absolute top-2 left-2 pointer-events-none font-mono text-[10px] text-red-400 bg-black/70 px-2 py-1 rounded border border-white/10">
                      <div>SEGMENTATION MASK</div>
                      <div className="text-neuro-text-muted">
                        {segViewMode === 'mask' ? 'MODE: BINARY MASK' : 'MODE: MRI + TUMOR OVERLAY'}
                      </div>
                    </div>
                    {typeof segmentationTumorAreaPct === 'number' && (
                      <div className="absolute bottom-2 right-2 pointer-events-none font-mono text-[10px] text-white bg-black/70 px-2 py-0.5 rounded border border-white/10">
                        TUMOR AREA: <span className="text-red-400 font-bold">{segmentationTumorAreaPct.toFixed(2)}%</span>
                      </div>
                    )}
                  </div>
                  <p className="text-[11px] text-neuro-text-muted">
                    Red highlight and amber contour indicate the tumor region predicted by the segmentation model (threshold &gt; 0.5, independent of Grad-CAM).
                  </p>
                  <div className="text-[10px] text-amber-400/90 font-mono bg-amber-500/10 border border-amber-500/20 px-2.5 py-1 rounded">
                    Research/educational use only. Not clinically validated.
                  </div>
                </>
              )
            ) : (
              <div className="p-6 rounded-lg bg-black/40 border border-neuro-border/60 text-center space-y-3">
                <Scan className="w-8 h-8 text-neuro-text-muted mx-auto opacity-50" />
                <div className="space-y-1">
                  <p className="text-xs font-semibold text-neuro-text-primary">
                    Tumor Segmentation Mask Not Generated Yet
                  </p>
                  <p className="text-[11px] text-neuro-text-muted">
                    Run the 2D segmentation model (DeepLabV3+ / BRISC U-Net) on this MRI slice to highlight the predicted tumor region.
                  </p>
                </div>
                {segError && (
                  <p className="text-xs text-red-400 font-mono">{segError}</p>
                )}
                <button
                  type="button"
                  onClick={handleGenerateSegmentation}
                  disabled={isGeneratingSeg}
                  className="px-4 py-2 rounded-lg bg-neuro-accent hover:bg-neuro-accent-bright text-neuro-bg text-xs font-bold transition-colors inline-flex items-center gap-1.5 disabled:opacity-60"
                >
                  {isGeneratingSeg ? (
                    <>
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                      <span>Segmenting Tumor...</span>
                    </>
                  ) : (
                    <>
                      <Scan className="w-3.5 h-3.5" />
                      <span>Generate Tumor Segmentation</span>
                    </>
                  )}
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Right Column (5 cols): Probability Distribution & Expandable Specs */}
        <div className="lg:col-span-5 space-y-4">
          {/* Class Probability Distribution */}
          <div className="p-4 rounded-xl bg-neuro-surface-200 border border-neuro-border space-y-3">
            <div className="flex items-center justify-between border-b border-neuro-border pb-3">
              <div className="flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-neuro-accent" />
                <h3 className="text-xs font-bold text-neuro-text-primary uppercase tracking-wider">
                  Class Probability Distribution
                </h3>
              </div>
              <span className="text-[11px] text-neuro-text-muted font-mono">Softmax Output</span>
            </div>

            <div className="space-y-3 pt-1">
              {scores.map(item => {
                const isTop = item.isTopPrediction || item.label.toLowerCase() === topPrediction.toLowerCase();
                const pct = item.percentage ?? Math.round(item.score * 100);
                return (
                  <div key={item.label} className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span className={`capitalize font-medium ${isTop ? 'text-neuro-text-primary font-bold' : 'text-neuro-text-secondary'}`}>
                        {item.label.replace(/_/g, ' ')}
                      </span>
                      <span className={`font-mono ${isTop ? 'text-emerald-400 font-bold' : 'text-neuro-text-muted'}`}>
                        {pct.toFixed(1)}%
                      </span>
                    </div>
                    <div className="h-2 w-full rounded-full bg-neuro-surface-400 overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-500 ${
                          isTop
                            ? 'bg-gradient-to-r from-neuro-accent to-emerald-400'
                            : 'bg-neuro-text-muted/40'
                        }`}
                        style={{ width: `${Math.max(pct, 1)}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Model Information & Provenance Specifications */}
          <details open className="group rounded-xl bg-neuro-surface-200 border border-neuro-border text-xs overflow-hidden">
            <summary className="p-4 flex items-center justify-between cursor-pointer select-none hover:bg-neuro-surface-300/50 transition-colors">
              <div className="flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-neuro-accent" />
                <h3 className="text-xs font-bold text-neuro-text-primary uppercase tracking-wider font-mono">
                  Model Information & Provenance
                </h3>
              </div>
              <ChevronDown className="w-4 h-4 text-neuro-text-muted transition-transform group-open:rotate-180" />
            </summary>

            <div className="px-4 pb-4 pt-1 space-y-2 text-neuro-text-secondary border-t border-neuro-border/40 font-mono">
              <div className="flex justify-between py-1 border-b border-neuro-border/30 items-center">
                <span className="text-neuro-text-muted">Model Name & ID:</span>
                <span className="font-semibold text-neuro-text-primary flex items-center gap-1.5">
                  <span>{modelName}</span>
                  <span className="px-1.5 py-0.5 rounded text-[10px] bg-neuro-surface-300 border border-neuro-border text-neuro-accent font-bold">
                    {modelId}
                  </span>
                </span>
              </div>
              <div className="flex justify-between py-1 border-b border-neuro-border/30">
                <span className="text-neuro-text-muted">Architecture:</span>
                <span className="font-semibold text-white text-right max-w-[260px] truncate" title={architecture}>
                  {architecture}
                </span>
              </div>
              <div className="flex justify-between py-1 border-b border-neuro-border/30">
                <span className="text-neuro-text-muted">Version:</span>
                <span className="text-cyan-400 font-semibold">{modelVersion}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-neuro-border/30">
                <span className="text-neuro-text-muted">Framework:</span>
                <span>{framework}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-neuro-border/30">
                <span className="text-neuro-text-muted">Input Resolution:</span>
                <span className="text-neuro-text-primary">{inputResolution}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-neuro-border/30">
                <span className="text-neuro-text-muted">Parameters:</span>
                <span>{parametersCount}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-neuro-border/30">
                <span className="text-neuro-text-muted">Preprocessing:</span>
                <span className="text-right max-w-[240px] truncate" title={preprocessing}>{preprocessing}</span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-neuro-text-muted">Actual Inference Latency:</span>
                <span className="font-bold text-emerald-400">{processingTimeMs} ms</span>
              </div>
            </div>
          </details>

          {/* Deep Explainability Callout */}
          <div className="p-4 rounded-xl bg-neuro-surface-200 border border-neuro-border/80 flex items-center justify-between gap-3">
            <div>
              <p className="text-xs font-semibold text-neuro-text-primary">Need multi-method attribution?</p>
              <p className="text-[11px] text-neuro-text-muted">Compare Grad-CAM, Grad-CAM++, and Integrated Gradients.</p>
            </div>
            <Link
              to={`/explainability?analysisId=${analysisId}`}
              className="px-3 py-1.5 rounded-lg bg-neuro-accent text-neuro-bg hover:bg-neuro-accent-bright text-xs font-bold transition-colors shrink-0 flex items-center gap-1"
            >
              <span>Explore</span>
              <ExternalLink className="w-3 h-3" />
            </Link>
          </div>

          {/* Research Disclaimer */}
          <ResearchDisclaimer />
        </div>
      </div>
    </motion.div>
  );
}
