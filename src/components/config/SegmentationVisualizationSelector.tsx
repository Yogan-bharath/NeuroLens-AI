import { Layers, Info, SplitSquareVertical, Eye, Image } from 'lucide-react';

interface SegmentationVisualizationSelectorProps {
  selectedVis: 'mask' | 'overlay' | 'comparison';
  onSelectVis: (vis: 'mask' | 'overlay' | 'comparison') => void;
  opacity: number;
  onChangeOpacity: (opacity: number) => void;
}

const visOptions: {
  id: 'mask' | 'overlay' | 'comparison';
  title: string;
  desc: string;
  icon: typeof Eye;
}[] = [
  {
    id: 'mask',
    title: 'Predicted Mask Only',
    desc: 'Render binary/multi-class semantic segmentation masks independently on neutral canvas.',
    icon: Layers,
  },
  {
    id: 'overlay',
    title: 'Original MRI + Mask Overlay',
    desc: 'Alpha-blend segmented tumor core and peritumoral edema directly on the original scan.',
    icon: Image,
  },
  {
    id: 'comparison',
    title: 'Side-by-Side Verification',
    desc: 'Dual-viewport comparing unenhanced scan against segmented structural masks.',
    icon: SplitSquareVertical,
  },
];

export default function SegmentationVisualizationSelector({
  selectedVis,
  onSelectVis,
  opacity,
  onChangeOpacity,
}: SegmentationVisualizationSelectorProps) {
  return (
    <div className="space-y-4 neuro-card p-4">
      {/* Section A: Segmentation Mask Visualization */}
      <div className="space-y-3">
        <div className="flex items-center justify-between border-b border-neuro-border pb-2.5">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-neuro-accent" />
            <h4 className="text-xs font-bold text-neuro-text-primary uppercase tracking-wider">
              Segmentation Mask Visualization (Model Output)
            </h4>
          </div>
          <span className="text-[10px] text-neuro-text-muted">
            Display Mode: <strong className="text-neuro-text-secondary capitalize">{selectedVis}</strong>
          </span>
        </div>

        <p className="text-xs text-neuro-text-secondary">
          Configure how the predicted structural segmentation boundaries will be rendered in the MRI viewer.
          <span className="text-neuro-text-muted block mt-0.5 text-[11px]">
            Note: This controls display visualization of the model prediction and is distinct from attribution XAI.
          </span>
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5 pt-1">
          {visOptions.map(opt => {
            const isSelected = selectedVis === opt.id;
            const Icon = opt.icon;

            return (
              <button
                key={opt.id}
                type="button"
                onClick={() => onSelectVis(opt.id)}
                className={`p-3 rounded-lg border text-left transition-all ${
                  isSelected
                    ? 'border-neuro-accent bg-neuro-accent-muted/20 text-neuro-accent-bright ring-1 ring-neuro-accent'
                    : 'border-neuro-border bg-neuro-surface-200 text-neuro-text-secondary hover:border-neuro-border-hover hover:text-neuro-text-primary'
                }`}
              >
                <div className="flex items-center gap-2 mb-1.5">
                  <Icon className="w-4 h-4" />
                  <span className="text-xs font-semibold">{opt.title}</span>
                </div>
                <p className="text-[11px] text-neuro-text-muted leading-relaxed">
                  {opt.desc}
                </p>
              </button>
            );
          })}
        </div>

        {/* Opacity slider */}
        {selectedVis !== 'mask' && (
          <div className="flex items-center justify-between p-2.5 rounded bg-neuro-surface-200/80 border border-neuro-border text-xs mt-2">
            <span className="text-neuro-text-secondary">Initial Mask Overlay Transparency:</span>
            <div className="flex items-center gap-2">
              <input
                type="range"
                min="0.2"
                max="1.0"
                step="0.05"
                value={opacity}
                onChange={e => onChangeOpacity(parseFloat(e.target.value))}
                className="w-24 accent-neuro-accent h-1.5 bg-neuro-surface-400 rounded"
              />
              <span className="font-mono text-neuro-text-primary w-8 text-right font-medium">
                {Math.round(opacity * 100)}%
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Section B: Segmentation-Specific XAI Status */}
      <div className="pt-3 border-t border-neuro-border">
        <div className="flex items-start gap-2.5 p-3 rounded-lg bg-neuro-surface-200 border border-neuro-border text-xs">
          <Info className="w-4 h-4 text-neuro-cyan flex-shrink-0 mt-0.5" />
          <div className="space-y-1">
            <h5 className="font-semibold text-neuro-text-primary text-[11px] uppercase tracking-wider">
              Segmentation-Specific XAI Attribution
            </h5>
            <p className="text-neuro-text-secondary leading-relaxed">
              Segmentation mask visualization is available. Additional segmentation explanation methods (such as pixel-level Grad-CAM or mask-conditioned Integrated Gradients) are not yet configured in this registry.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
