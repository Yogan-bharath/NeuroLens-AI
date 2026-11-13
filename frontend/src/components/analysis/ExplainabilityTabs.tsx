import type { ExplanationResult, SegmentationResult } from '@/types/analysis';
import HeatmapOverlayViewer from '../viewer/HeatmapOverlayViewer';
import SegmentationViewer from '../viewer/SegmentationViewer';
import { Sparkles, Layers, Info, SplitSquareVertical } from 'lucide-react';

export type ActiveTabType = 'gradcam' | 'gradcam_pp' | 'segmentation' | 'comparison';

interface ExplainabilityTabsProps {
  activeTab: ActiveTabType;
  onChangeTab: (tab: ActiveTabType) => void;
  explanations: ExplanationResult[];
  segmentation: SegmentationResult | null;
  targetClass: string;
}

export default function ExplainabilityTabs({
  activeTab,
  onChangeTab,
  explanations,
  segmentation,
  targetClass,
}: ExplainabilityTabsProps) {
  const gradcamExp = explanations.find(e => e.method === 'gradcam');
  const gradcamPPExp = explanations.find(e => e.method === 'gradcam_pp');

  return (
    <div className="neuro-card space-y-4">
      {/* Tabs Header */}
      <div className="flex items-center justify-between border-b border-neuro-border pb-3">
        <div className="flex items-center gap-1.5">
          <Sparkles className="w-4 h-4 text-neuro-accent" />
          <h3 className="text-sm font-semibold text-neuro-text-primary">
            Explainable AI (XAI)
          </h3>
        </div>
        <span className="text-xs text-neuro-text-muted">Target: {targetClass}</span>
      </div>

      {/* Tab Buttons */}
      <div className="grid grid-cols-4 gap-1.5 p-1 bg-neuro-surface-200 rounded-lg text-xs">
        <button
          onClick={() => onChangeTab('gradcam')}
          className={`px-2.5 py-1.5 rounded-md font-medium transition-all ${
            activeTab === 'gradcam'
              ? 'bg-neuro-accent text-neuro-bg shadow-sm font-semibold'
              : 'text-neuro-text-secondary hover:text-neuro-text-primary hover:bg-neuro-surface-300'
          }`}
        >
          Grad-CAM
        </button>
        <button
          onClick={() => onChangeTab('gradcam_pp')}
          className={`px-2.5 py-1.5 rounded-md font-medium transition-all ${
            activeTab === 'gradcam_pp'
              ? 'bg-neuro-accent text-neuro-bg shadow-sm font-semibold'
              : 'text-neuro-text-secondary hover:text-neuro-text-primary hover:bg-neuro-surface-300'
          }`}
        >
          Grad-CAM++
        </button>
        <button
          onClick={() => onChangeTab('segmentation')}
          className={`px-2.5 py-1.5 rounded-md font-medium transition-all ${
            activeTab === 'segmentation'
              ? 'bg-neuro-accent text-neuro-bg shadow-sm font-semibold'
              : 'text-neuro-text-secondary hover:text-neuro-text-primary hover:bg-neuro-surface-300'
          }`}
        >
          Segmentation
        </button>
        <button
          onClick={() => onChangeTab('comparison')}
          className={`px-2.5 py-1.5 rounded-md font-medium transition-all flex items-center justify-center gap-1 ${
            activeTab === 'comparison'
              ? 'bg-neuro-accent text-neuro-bg shadow-sm font-semibold'
              : 'text-neuro-text-secondary hover:text-neuro-text-primary hover:bg-neuro-surface-300'
          }`}
        >
          <SplitSquareVertical className="w-3 h-3" />
          <span>Compare</span>
        </button>
      </div>

      {/* Tab Content Details */}
      <div className="space-y-3">
        {activeTab === 'gradcam' && (
          <div className="space-y-3 animate-fade-in">
            <HeatmapOverlayViewer
              methodName="Grad-CAM"
              targetClass={targetClass}
              layerName={gradcamExp?.layerName || 'layer4 (ResNet-50)'}
              description={
                gradcamExp?.description ||
                'Highlights coarse regions that positively influenced the prediction for this target class.'
              }
            />
            <div className="p-2.5 rounded bg-neuro-surface-200/60 border border-neuro-border text-[11px] text-neuro-text-muted">
              <strong className="text-neuro-text-secondary font-medium">Important distinction:</strong> Grad-CAM highlights decision influence areas; it does NOT denote anatomical tumor boundaries or margin extents.
            </div>
          </div>
        )}

        {activeTab === 'gradcam_pp' && (
          <div className="space-y-3 animate-fade-in">
            <HeatmapOverlayViewer
              methodName="Grad-CAM++"
              targetClass={targetClass}
              layerName={gradcamPPExp?.layerName || 'layer4 (ResNet-50)'}
              description={
                gradcamPPExp?.description ||
                'Grad-CAM++ provides pixel-weighted combinations to capture multiple salient occurrences with finer spatial localization.'
              }
            />
            <div className="p-2.5 rounded bg-neuro-surface-200/60 border border-neuro-border text-[11px] text-neuro-text-muted">
              Grad-CAM++ uses higher-order partial derivatives to provide clearer object localization when multiple instances or diffused anomalies are present.
            </div>
          </div>
        )}

        {activeTab === 'segmentation' && (
          <div className="space-y-3 animate-fade-in">
            {/* Segmentation XAI Layer Indicator */}
            <div className="p-3 rounded-lg bg-neuro-surface-200/80 border border-neuro-border text-xs space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5 font-semibold text-neuro-cyan">
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>Segmentation XAI Layer: bottleneck.conv2</span>
                </div>
                <span className="neuro-badge-accent text-[10px]">U-Net Bottleneck</span>
              </div>
              <p className="text-[11px] text-neuro-text-secondary leading-relaxed">
                Backpropagated gradient saliency confirms that deepest 512-channel latent feature maps predominantly governed the delineation of tumor core margins vs. peritumoral edema.
              </p>
              <div className="flex items-center gap-2 pt-1 border-t border-neuro-border/60 text-[11px] text-neuro-text-muted">
                <SplitSquareVertical className="w-3.5 h-3.5 text-neuro-cyan" />
                <span>Layout: <strong className="text-neuro-text-primary">Side-by-Side (Active by Default)</strong></span>
              </div>
            </div>

            {segmentation ? (
              <SegmentationViewer
                classes={segmentation.classes}
                description={segmentation.description}
              />
            ) : (
              <div className="p-6 text-center text-xs text-neuro-text-muted bg-neuro-surface-200 rounded-lg">
                Segmentation mask not available for this analysis.
              </div>
            )}
            <div className="p-2.5 rounded bg-neuro-surface-200/60 border border-neuro-border text-[11px] text-neuro-text-muted flex items-start gap-1.5">
              <Info className="w-3.5 h-3.5 text-neuro-cyan flex-shrink-0 mt-0.5" />
              <p>
                The predicted segmentation mask represents pixel-wise spatial classification produced by the U-Net expanding path and verified via the bottleneck XAI layer.
              </p>
            </div>
          </div>
        )}

        {activeTab === 'comparison' && (
          <div className="space-y-3 animate-fade-in">
            <div className="p-3 bg-neuro-surface-200 rounded-lg border border-neuro-border text-xs space-y-2">
              <div className="flex items-center gap-1.5 font-medium text-neuro-text-primary">
                <SplitSquareVertical className="w-3.5 h-3.5 text-neuro-accent" />
                <span>Side-by-Side Verification Mode</span>
              </div>
              <p className="text-[11px] text-neuro-text-secondary leading-relaxed">
                Compare the original unenhanced scan with the active explainability overlay in the viewer above. Use the viewer's "Side-by-Side" button to toggle synchronous dual-canvas inspection.
              </p>
              <div className="grid grid-cols-2 gap-2 pt-2 border-t border-neuro-border">
                <div className="p-2 rounded bg-neuro-surface-300/50">
                  <p className="text-[10px] text-neuro-text-muted uppercase font-semibold">Left Canvas</p>
                  <p className="text-xs text-neuro-text-primary mt-0.5">Original MRI Scan</p>
                </div>
                <div className="p-2 rounded bg-neuro-surface-300/50">
                  <p className="text-[10px] text-neuro-text-muted uppercase font-semibold">Right Canvas</p>
                  <p className="text-xs text-neuro-accent-bright mt-0.5">XAI Layer Overlay</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
