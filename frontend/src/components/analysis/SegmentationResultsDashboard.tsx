import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Download,
  Layers,
  Activity,
  Sparkles,
  Eye,
  Sliders,
  Maximize2,
  AlertTriangle,
  CheckCircle2,
  FileCheck,
  Brain,
  Info,
} from 'lucide-react';
import Tumor3DViewer from '@/components/viewer/Tumor3DViewer';
import type { SegmentationAnalysisResponse } from '@/services/api';

interface Props {
  data: SegmentationAnalysisResponse;
  onReset?: () => void;
}

export default function SegmentationResultsDashboard({ data, onReset }: Props) {
  const [activePlane, setActivePlane] = useState<'axial' | 'coronal' | 'sagittal'>('axial');
  const [overlayOpacity, setOverlayOpacity] = useState<number>(0.65);
  const [activeExplainTab, setActiveExplainTab] = useState<'gradcam' | 'bottleneck' | 'ablation' | 'uncertainty'>('gradcam');
  const [selectedCamClass, setSelectedCamClass] = useState<string>('Enhancing Tumor');

  const sliceData = data.slices[activePlane];
  const totalTumorVolume = data.class_distribution
    .filter(c => c.index > 0)
    .reduce((acc, c) => acc + c.volume_mm3, 0);

  const downloadUrl = data.download_url;

  return (
    <div className="space-y-6">
      {/* Top Banner & Actions */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 p-4 rounded-xl bg-neuro-surface-200 border border-neuro-border">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
            <CheckCircle2 className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <h2 className="text-base font-bold text-neuro-text-primary">3D Segmentation Completed</h2>
              {data.analysis_id && (
                <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-semibold border border-emerald-500/30">
                  {data.analysis_id} · Saved to Database
                </span>
              )}
            </div>
            <p className="text-xs text-neuro-text-secondary mt-0.5">
              Processed 4 multimodal NIfTI volumes in {data.processing_time_ms} ms · Dimensions: {data.volume_dimensions.join(' × ')}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2 w-full sm:w-auto">
          <a
            href={downloadUrl}
            download
            className="flex-1 sm:flex-initial flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-neuro-accent hover:bg-neuro-accent-bright text-white text-xs font-semibold shadow-sm transition-all"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Download NIfTI (.nii.gz)</span>
          </a>
          {onReset && (
            <button
              onClick={onReset}
              className="px-3 py-2 rounded-lg bg-neuro-surface-300 hover:bg-neuro-surface-400 text-neuro-text-secondary text-xs font-medium border border-neuro-border transition-colors"
            >
              New Scan
            </button>
          )}
        </div>
      </div>

      {/* Main Grid: Multi-Planar 2D View & 3D Interactive View */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: Multi-Planar 2D Slice Viewer */}
        <div className="flex flex-col rounded-xl overflow-hidden border border-neuro-border bg-neuro-surface-200">
          {/* Header Controls */}
          <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-3 border-b border-neuro-border bg-neuro-surface-300/60">
            <div className="flex items-center gap-1.5 bg-neuro-surface-400/50 p-1 rounded-lg">
              {(['axial', 'coronal', 'sagittal'] as const).map(plane => (
                <button
                  key={plane}
                  onClick={() => setActivePlane(plane)}
                  className={`px-3 py-1 rounded text-xs font-semibold capitalize transition-all ${
                    activePlane === plane
                      ? 'bg-neuro-accent text-white shadow-sm'
                      : 'text-neuro-text-secondary hover:text-neuro-text-primary'
                  }`}
                >
                  {plane}
                </button>
              ))}
            </div>

            <div className="flex items-center gap-2 text-xs">
              <span className="text-neuro-text-muted">Overlay Opacity:</span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={overlayOpacity}
                onChange={e => setOverlayOpacity(parseFloat(e.target.value))}
                className="w-20 h-1.5 bg-neuro-surface-400 rounded-lg appearance-none cursor-pointer accent-neuro-accent"
              />
              <span className="text-[11px] font-mono text-neuro-text-primary w-8">
                {Math.round(overlayOpacity * 100)}%
              </span>
            </div>
          </div>

          {/* 2D Slice Image Display */}
          <div className="relative aspect-square w-full max-h-[400px] flex items-center justify-center bg-black overflow-hidden">
            {sliceData ? (
              <img
                src={sliceData.preview}
                alt={`${activePlane} slice`}
                className="w-full h-full object-contain"
                style={{ opacity: 1 }}
              />
            ) : (
              <div className="text-xs text-neuro-text-muted">No slice preview available</div>
            )}

            {/* Slice index badge */}
            <div className="absolute top-3 left-3 px-2 py-1 rounded bg-black/60 backdrop-blur-sm border border-slate-700 text-[11px] text-white font-mono">
              {activePlane.toUpperCase()} Slice #{sliceData?.slice_index ?? 0}
            </div>

            {/* Color Legend Overlay */}
            <div className="absolute bottom-3 right-3 flex flex-col gap-1 p-2 rounded bg-black/75 backdrop-blur-sm border border-slate-700 text-[10px]">
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-[#EF4444]" />
                <span className="text-slate-200">Necrotic Core</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-[#10B981]" />
                <span className="text-slate-200">Edema</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-[#F59E0B]" />
                <span className="text-slate-200">Enhancing Tumor</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right: 3D Interactive Tumor Canvas */}
        <Tumor3DViewer surfaceData={data.surface_3d} dimensions={data.volume_dimensions} />
      </div>

      {/* Volumetric & Class Distribution Breakdown */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {data.class_distribution.map(cls => (
          <div
            key={cls.index}
            className="p-4 rounded-xl bg-neuro-surface-200 border border-neuro-border flex flex-col justify-between"
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-semibold text-neuro-text-secondary">{cls.name}</span>
              <span
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: cls.color === '#000000' ? '#64748B' : cls.color }}
              />
            </div>
            <div className="space-y-1">
              <div className="text-xl font-bold text-neuro-text-primary">
                {cls.percentage}%
              </div>
              <div className="text-xs text-neuro-text-muted flex items-center justify-between">
                <span>{cls.voxel_count.toLocaleString()} voxels</span>
                {cls.index > 0 && <span>{(cls.volume_mm3 / 1000).toFixed(2)} cm³</span>}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Ground-Truth Evaluation Metrics Table (If SEG provided) */}
      {data.evaluation_metrics && (
        <div className="p-5 rounded-xl bg-neuro-surface-200 border border-neuro-border space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <FileCheck className="w-4 h-4 text-emerald-400" />
              <h3 className="text-sm font-bold text-neuro-text-primary">Ground-Truth Evaluation Metrics</h3>
            </div>
            <span className="text-[11px] px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 font-medium border border-emerald-500/20">
              Evaluated against BraTS SEG mask
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-neuro-border text-neuro-text-muted font-semibold">
                  <th className="py-2 px-3">Anatomical Lesion Sub-Region</th>
                  <th className="py-2 px-3">Dice Coefficient</th>
                  <th className="py-2 px-3">Intersection over Union (IoU)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-neuro-border text-neuro-text-primary">
                <tr className="hover:bg-neuro-surface-300/40">
                  <td className="py-2 px-3 font-semibold text-neuro-accent-bright">Whole Tumor (WT: Labels 1, 2, 3)</td>
                  <td className="py-2 px-3 font-mono">{(data.evaluation_metrics.whole_tumor_dice * 100).toFixed(2)}%</td>
                  <td className="py-2 px-3 font-mono">{(data.evaluation_metrics.whole_tumor_iou * 100).toFixed(2)}%</td>
                </tr>
                <tr className="hover:bg-neuro-surface-300/40">
                  <td className="py-2 px-3 font-semibold text-neuro-cyan">Tumor Core (TC: Labels 1, 3)</td>
                  <td className="py-2 px-3 font-mono">{(data.evaluation_metrics.tumor_core_dice * 100).toFixed(2)}%</td>
                  <td className="py-2 px-3 font-mono">{(data.evaluation_metrics.tumor_core_iou * 100).toFixed(2)}%</td>
                </tr>
                <tr className="hover:bg-neuro-surface-300/40">
                  <td className="py-2 px-3 text-red-400">Necrotic / Non-enhancing (Label 1)</td>
                  <td className="py-2 px-3 font-mono">{(data.evaluation_metrics.necrotic_dice * 100).toFixed(2)}%</td>
                  <td className="py-2 px-3 font-mono">{(data.evaluation_metrics.necrotic_iou * 100).toFixed(2)}%</td>
                </tr>
                <tr className="hover:bg-neuro-surface-300/40">
                  <td className="py-2 px-3 text-emerald-400">Peritumoral Edema (Label 2)</td>
                  <td className="py-2 px-3 font-mono">{(data.evaluation_metrics.edema_dice * 100).toFixed(2)}%</td>
                  <td className="py-2 px-3 font-mono">{(data.evaluation_metrics.edema_iou * 100).toFixed(2)}%</td>
                </tr>
                <tr className="hover:bg-neuro-surface-300/40">
                  <td className="py-2 px-3 text-amber-400">Enhancing Tumor (ET: Label 3)</td>
                  <td className="py-2 px-3 font-mono">{(data.evaluation_metrics.enhancing_dice * 100).toFixed(2)}%</td>
                  <td className="py-2 px-3 font-mono">{(data.evaluation_metrics.enhancing_iou * 100).toFixed(2)}%</td>
                </tr>
                <tr className="bg-neuro-surface-300/30 font-semibold border-t-2 border-neuro-border">
                  <td className="py-2 px-3 text-neuro-text-primary">Mean Foreground Average</td>
                  <td className="py-2 px-3 font-mono text-neuro-accent-bright">
                    {(data.evaluation_metrics.mean_foreground_dice * 100).toFixed(2)}%
                  </td>
                  <td className="py-2 px-3 font-mono text-neuro-cyan">
                    {(data.evaluation_metrics.mean_foreground_iou * 100).toFixed(2)}%
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Explainability Studio */}
      <div className="p-5 rounded-xl bg-neuro-surface-200 border border-neuro-border space-y-4">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 border-b border-neuro-border pb-3">
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-neuro-accent" />
            <h3 className="text-sm font-bold text-neuro-text-primary">3D Explainable AI Studio (XAI)</h3>
          </div>

          <div className="flex items-center gap-1 bg-neuro-surface-300 p-1 rounded-lg text-xs">
            <button
              onClick={() => setActiveExplainTab('gradcam')}
              className={`px-3 py-1 rounded transition-all ${
                activeExplainTab === 'gradcam'
                  ? 'bg-neuro-accent text-white font-semibold'
                  : 'text-neuro-text-secondary hover:text-neuro-text-primary'
              }`}
            >
              3D Grad-CAM
            </button>
            <button
              onClick={() => setActiveExplainTab('bottleneck')}
              className={`px-3 py-1 rounded transition-all ${
                activeExplainTab === 'bottleneck'
                  ? 'bg-neuro-accent text-white font-semibold'
                  : 'text-neuro-text-secondary hover:text-neuro-text-primary'
              }`}
            >
              Bottleneck Features
            </button>
            <button
              onClick={() => setActiveExplainTab('ablation')}
              className={`px-3 py-1 rounded transition-all ${
                activeExplainTab === 'ablation'
                  ? 'bg-neuro-accent text-white font-semibold'
                  : 'text-neuro-text-secondary hover:text-neuro-text-primary'
              }`}
            >
              Modality Ablation
            </button>
            <button
              onClick={() => setActiveExplainTab('uncertainty')}
              className={`px-3 py-1 rounded transition-all ${
                activeExplainTab === 'uncertainty'
                  ? 'bg-neuro-accent text-white font-semibold'
                  : 'text-neuro-text-secondary hover:text-neuro-text-primary'
              }`}
            >
              Uncertainty (Entropy)
            </button>
          </div>
        </div>

        {/* Tab 1: 3D Grad-CAM */}
        {activeExplainTab === 'gradcam' && (
          <div className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
              <span className="text-neuro-text-secondary">
                Target Layer: <code className="text-neuro-accent-bright font-mono">{data.explainability.gradcam_layer || 'bottleneck_relu2'}</code>
              </span>
              <div className="flex items-center gap-1.5">
                {Object.keys(data.explainability.gradcam_maps || {}).map(clsName => (
                  <button
                    key={clsName}
                    onClick={() => setSelectedCamClass(clsName)}
                    className={`px-2.5 py-1 rounded-lg text-xs font-medium border transition-all ${
                      selectedCamClass === clsName
                        ? 'bg-neuro-accent text-white border-transparent'
                        : 'bg-neuro-surface-300 text-neuro-text-secondary border-neuro-border hover:bg-neuro-surface-400'
                    }`}
                  >
                    {clsName}
                  </button>
                ))}
              </div>
            </div>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-6 p-4 rounded-lg bg-black/40 border border-neuro-border">
              {data.explainability.gradcam_maps && data.explainability.gradcam_maps[selectedCamClass] ? (
                <div className="flex flex-col items-center gap-2">
                  <img
                    src={data.explainability.gradcam_maps[selectedCamClass]}
                    alt={`Grad-CAM ${selectedCamClass}`}
                    className="max-h-64 rounded-lg border border-slate-700 shadow-md"
                  />
                  <span className="text-[11px] text-neuro-text-muted">
                    Grad-CAM activation for {selectedCamClass} overlaid on FLAIR slice
                  </span>
                </div>
              ) : (
                <div className="text-xs text-neuro-text-muted py-8">
                  Select a class above to inspect its 3D Grad-CAM activation heatmap.
                </div>
              )}
            </div>
          </div>
        )}

        {/* Tab 2: Bottleneck Features */}
        {activeExplainTab === 'bottleneck' && (
          <div className="space-y-3">
            <p className="text-xs text-neuro-text-secondary">
              Activations extracted from layer <code className="text-neuro-accent-bright">bottleneck_relu2</code> (256 channels).
              Dimensions: {data.explainability.bottleneck_features?.shape?.join(' × ') || '12 × 12 × 8 × 256'}.
            </p>
            <div className="p-4 rounded-lg bg-neuro-surface-300/40 border border-neuro-border text-xs text-neuro-text-muted flex items-center gap-3">
              <Info className="w-4 h-4 text-neuro-cyan flex-shrink-0" />
              <span>
                Bottleneck layer captures high-level latent representations uniting information from all four multimodal channels prior to decoder upsampling.
              </span>
            </div>
          </div>
        )}

        {/* Tab 3: Modality Ablation */}
        {activeExplainTab === 'ablation' && (
          <div className="space-y-3">
            <p className="text-xs text-neuro-text-secondary">
              Evaluates the model's sensitivity when individual MRI modalities are removed (zeroed out):
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
              {data.explainability.modality_ablation?.map(item => (
                <div key={item.removed_modality} className="p-3 rounded-lg bg-neuro-surface-300/50 border border-neuro-border">
                  <div className="flex items-center justify-between text-xs mb-1">
                    <span className="font-semibold text-neuro-text-primary">Without {item.removed_modality}</span>
                    <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${
                      item.impact_level === 'High'
                        ? 'bg-red-500/10 text-red-400'
                        : item.impact_level === 'Moderate'
                        ? 'bg-amber-500/10 text-amber-400'
                        : 'bg-emerald-500/10 text-emerald-400'
                    }`}>
                      {item.impact_level} Impact
                    </span>
                  </div>
                  <div className="text-lg font-bold text-neuro-text-primary">
                    {item.relative_change_pct > 0 ? `+${item.relative_change_pct}%` : `${item.relative_change_pct}%`}
                  </div>
                  <div className="text-[11px] text-neuro-text-muted">
                    {item.tumor_voxel_count.toLocaleString()} voxels detected
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Tab 4: Uncertainty (Entropy) */}
        {activeExplainTab === 'uncertainty' && (
          <div className="space-y-3">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="p-4 rounded-lg bg-neuro-surface-300/50 border border-neuro-border space-y-1">
                <span className="text-xs text-neuro-text-muted">Overall Volume Normalized Entropy</span>
                <div className="text-xl font-bold text-neuro-cyan">
                  {data.explainability.uncertainty?.mean_entropy_overall ?? 'N/A'}
                </div>
                <p className="text-[11px] text-neuro-text-muted">
                  Mean Shannon entropy across all brain voxels normalized to [0, 1].
                </p>
              </div>

              <div className="p-4 rounded-lg bg-neuro-surface-300/50 border border-neuro-border space-y-1">
                <span className="text-xs text-neuro-text-muted">Lesion Boundary Uncertainty</span>
                <div className="text-xl font-bold text-amber-400">
                  {data.explainability.uncertainty?.mean_entropy_tumor ?? 'N/A'}
                </div>
                <p className="text-[11px] text-neuro-text-muted">
                  Predictive entropy concentrated at the infiltration boundary.
                </p>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Research Disclaimer */}
      <div className="p-4 rounded-xl bg-amber-500/5 border border-amber-500/20 text-xs text-amber-300 flex items-start gap-3">
        <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
        <div>
          <p className="font-semibold text-amber-200">Investigational Research Model Output</p>
          <p className="text-amber-300/90 mt-0.5">{data.disclaimer}</p>
        </div>
      </div>
    </div>
  );
}

