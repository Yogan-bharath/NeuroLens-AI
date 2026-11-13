import { useState } from 'react';
import type { FeatureLayerOption, XaiMethod } from '@/types/modelRegistry';
import { Sliders, ChevronDown, ChevronUp, CheckCircle2, AlertCircle } from 'lucide-react';

interface AdvancedExplainabilitySettingsProps {
  layers: FeatureLayerOption[];
  selectedLayerId: string;
  onSelectLayer: (layerId: string) => void;
  xaiMethod: XaiMethod | undefined;
  modelName: string;
}

export default function AdvancedExplainabilitySettings({
  layers,
  selectedLayerId,
  onSelectLayer,
  xaiMethod,
  modelName,
}: AdvancedExplainabilitySettingsProps) {
  const [isOpen, setIsOpen] = useState(false);

  // Hidden if XAI method does not support layer selection (e.g. Integrated Gradients)
  if (!xaiMethod?.supportsLayerSelection) {
    return null;
  }

  const selectedLayer = layers.find(l => l.id === selectedLayerId) || layers[0];

  return (
    <div className="neuro-card p-0 border border-neuro-border overflow-hidden">
      {/* Collapsible Header */}
      <button
        type="button"
        onClick={() => setIsOpen(prev => !prev)}
        className="w-full flex items-center justify-between p-4 bg-neuro-surface-200/50 hover:bg-neuro-surface-200 transition-colors text-left"
      >
        <div className="flex items-center gap-2">
          <Sliders className="w-4 h-4 text-neuro-accent" />
          <div>
            <h4 className="text-xs font-bold text-neuro-text-primary uppercase tracking-wider">
              Advanced Explainability Settings
            </h4>
            <p className="text-[11px] text-neuro-text-muted mt-0.5">
              Target Convolutional Layer Attribution for {xaiMethod.name}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-neuro-accent-bright hidden sm:inline">
            {selectedLayer?.name || 'Default Layer'}
          </span>
          {isOpen ? <ChevronUp className="w-4 h-4 text-neuro-text-muted" /> : <ChevronDown className="w-4 h-4 text-neuro-text-muted" />}
        </div>
      </button>

      {/* Collapsible Body */}
      {isOpen && (
        <div className="p-4 space-y-4 border-t border-neuro-border bg-neuro-surface text-xs animate-fade-in">
          <div>
            <label className="block text-neuro-text-primary font-medium mb-1">
              Convolutional Feature Layer ({modelName}):
            </label>
            <p className="text-[11px] text-neuro-text-muted mb-2">
              Gradients will be extracted and pooled across this specific tensor depth to formulate the saliency heatmap.
            </p>

            <select
              value={selectedLayerId}
              onChange={e => onSelectLayer(e.target.value)}
              className="neuro-input w-full text-xs font-mono py-2"
            >
              {layers.map(layer => (
                <option key={layer.id} value={layer.id}>
                  {layer.name} {layer.isVerified ? '(Verified & Recommended)' : '(Experimental Layer)'}
                </option>
              ))}
            </select>
          </div>

          {selectedLayer && (
            <div className="p-3 rounded-lg bg-neuro-surface-200 border border-neuro-border space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="font-semibold text-neuro-text-primary font-mono">{selectedLayer.name}</span>
                {selectedLayer.isVerified ? (
                  <span className="neuro-badge-success text-[10px] flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3" />
                    Verified Target
                  </span>
                ) : (
                  <span className="neuro-badge-warning text-[10px] flex items-center gap-1">
                    <AlertCircle className="w-3 h-3" />
                    Experimental
                  </span>
                )}
              </div>
              <p className="text-neuro-text-secondary text-[11px] leading-relaxed">
                {selectedLayer.description}
              </p>
              {selectedLayer.depthDescription && (
                <p className="text-neuro-text-muted text-[10px] font-mono">
                  {selectedLayer.depthDescription}
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
