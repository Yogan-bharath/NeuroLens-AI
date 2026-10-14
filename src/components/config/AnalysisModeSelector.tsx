import type { AnalysisMode } from '@/types/modelRegistry';
import { Brain, Layers, Cpu, Check } from 'lucide-react';

interface AnalysisModeSelectorProps {
  selectedMode: AnalysisMode;
  onSelectMode: (mode: AnalysisMode) => void;
}

const modes: {
  id: AnalysisMode;
  title: string;
  tagline: string;
  description: string;
  icon: typeof Brain;
}[] = [
  {
    id: 'classification',
    title: 'Classification Only',
    tagline: 'Pathological Typing',
    description: 'Predict tumor category (Glioma, Meningioma, Pituitary, or No Tumor) and generate classification saliency heatmaps.',
    icon: Brain,
  },
  {
    id: 'segmentation',
    title: 'Segmentation Only',
    tagline: 'Structural Delineation',
    description: 'Perform pixel-level semantic segmentation identifying Tumor Core and Peritumoral Edema spatial boundaries.',
    icon: Layers,
  },
  {
    id: 'combined',
    title: 'Combined Analysis',
    tagline: 'Dual Pipeline (Recommended)',
    description: 'Execute both classification and segmentation in parallel with distinct, dedicated explainability configurations.',
    icon: Cpu,
  },
];

export default function AnalysisModeSelector({
  selectedMode,
  onSelectMode,
}: AnalysisModeSelectorProps) {
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-neuro-text-primary uppercase tracking-wider text-xs">
          Step 1: Choose Analysis Mode
        </h3>
        <span className="text-xs text-neuro-text-muted">
          Active Mode: <strong className="text-neuro-accent-bright capitalize">{selectedMode}</strong>
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {modes.map(mode => {
          const isSelected = selectedMode === mode.id;
          const Icon = mode.icon;

          return (
            <button
              key={mode.id}
              type="button"
              onClick={() => onSelectMode(mode.id)}
              className={`neuro-card text-left p-4 transition-all duration-200 relative cursor-pointer border flex flex-col justify-between ${
                isSelected
                  ? 'border-neuro-accent bg-neuro-accent-muted/30 shadow-neuro-glow'
                  : 'border-neuro-border bg-neuro-surface hover:border-neuro-border-hover'
              }`}
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <div
                    className={`w-9 h-9 rounded-lg flex items-center justify-center ${
                      isSelected
                        ? 'bg-neuro-accent text-neuro-bg'
                        : 'bg-neuro-surface-200 text-neuro-text-secondary'
                    }`}
                  >
                    <Icon className="w-5 h-5" />
                  </div>
                  {isSelected && (
                    <div className="w-5 h-5 rounded-full bg-neuro-accent text-neuro-bg flex items-center justify-center">
                      <Check className="w-3.5 h-3.5 stroke-[3]" />
                    </div>
                  )}
                </div>

                <h4 className="text-sm font-semibold text-neuro-text-primary">{mode.title}</h4>
                <p className="text-[11px] font-medium text-neuro-accent-bright mt-0.5">{mode.tagline}</p>
                <p className="text-xs text-neuro-text-secondary mt-2 leading-relaxed">
                  {mode.description}
                </p>
              </div>

              <div className="pt-3 mt-3 border-t border-neuro-border/60 flex items-center justify-between text-[11px]">
                <span className="text-neuro-text-muted">Pipeline scope:</span>
                <span className="font-mono text-neuro-text-secondary">
                  {mode.id === 'combined' ? 'Class + Seg' : mode.id === 'classification' ? 'Class Only' : 'Seg Only'}
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
