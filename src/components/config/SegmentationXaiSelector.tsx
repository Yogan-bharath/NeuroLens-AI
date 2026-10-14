import type { XaiMethod, FeatureLayerOption, SegmentationModel } from '@/types/modelRegistry';
import { Layers, Sparkles, Check, Sliders, SplitSquareVertical, CheckCircle2 } from 'lucide-react';

interface SegmentationXaiSelectorProps {
  methods: XaiMethod[];
  selectedMethodId: string;
  onSelectMethod: (methodId: string) => void;
  layers: FeatureLayerOption[];
  selectedLayerId: string;
  onSelectLayer: (layerId: string) => void;
  segmentationModel: SegmentationModel | undefined;
}

export default function SegmentationXaiSelector({
  methods,
  selectedMethodId,
  onSelectMethod,
  layers,
  selectedLayerId,
  onSelectLayer,
  segmentationModel,
}: SegmentationXaiSelectorProps) {
  const selectedMethod = methods.find(m => m.id === selectedMethodId) || methods[0];
  const selectedLayer = layers.find(l => l.id === selectedLayerId) || layers[0];
  const modelName = segmentationModel?.name || 'U-Net';

  return (
    <div className="space-y-4">
      {/* Header and Default Side-by-Side Notice */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div>
          <h3 className="text-sm font-semibold text-neuro-text-primary uppercase tracking-wider text-xs">
            Step 3 (B): Segmentation Explainable AI (XAI) & Layer
          </h3>
          <p className="text-xs text-neuro-text-muted mt-0.5">
            Configure latent feature attribution targeting <strong className="text-neuro-cyan">{modelName}</strong> boundary predictions.
          </p>
        </div>

        {/* Default side-by-side badge */}
        <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-neuro-surface-200 border border-neuro-border text-xs text-neuro-text-secondary w-fit">
          <SplitSquareVertical className="w-3.5 h-3.5 text-neuro-cyan" />
          <span>Visualization: <strong className="text-neuro-text-primary">Side-by-Side (Default)</strong></span>
        </div>
      </div>

      {/* Segmentation XAI Method Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {methods.map(method => {
          const isSelected = selectedMethodId === method.id;
          const isCompatible = segmentationModel?.supportedXaiMethods.includes(method.id) ?? true;

          return (
            <div
              key={method.id}
              onClick={() => isCompatible && onSelectMethod(method.id)}
              className={`neuro-card p-4 transition-all duration-200 border flex flex-col justify-between relative ${
                isCompatible ? 'cursor-pointer' : 'cursor-not-allowed opacity-60'
              } ${
                isSelected
                  ? 'border-neuro-cyan bg-neuro-cyan-muted/20 shadow-neuro-glow ring-1 ring-neuro-cyan'
                  : isCompatible
                  ? 'border-neuro-border bg-neuro-surface hover:border-neuro-border-hover hover:bg-neuro-surface-200/40'
                  : 'border-neuro-border/40 bg-neuro-surface/40'
              }`}
            >
              <div>
                <div className="flex items-start justify-between gap-2 mb-2">
                  <div className="flex items-center gap-2">
                    <div
                      className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                        isSelected
                          ? 'bg-neuro-cyan text-neuro-bg'
                          : 'bg-neuro-surface-200 text-neuro-text-secondary'
                      }`}
                    >
                      <Sparkles className="w-4 h-4" />
                    </div>
                    <div>
                      <h4 className="text-sm font-bold text-neuro-text-primary">{method.name}</h4>
                      <span className="text-[10px] text-neuro-text-muted uppercase font-mono">
                        {method.type.replace('_', ' ')}
                      </span>
                    </div>
                  </div>

                  {isSelected && (
                    <div className="w-5 h-5 rounded-full bg-neuro-cyan text-neuro-bg flex items-center justify-center">
                      <Check className="w-3.5 h-3.5 stroke-[3]" />
                    </div>
                  )}
                </div>

                <p className="text-xs text-neuro-text-secondary leading-relaxed mt-2">
                  {method.description}
                </p>
              </div>

              <div className="pt-3 mt-4 border-t border-neuro-border/60 flex items-center justify-between text-xs">
                <span className="text-[11px] text-neuro-text-muted">
                  {method.supportsLayerSelection ? 'Encoder/Decoder Layer Tuning' : 'Direct Pixel Attribution'}
                </span>

                <button
                  type="button"
                  disabled={!isCompatible}
                  onClick={e => {
                    e.stopPropagation();
                    if (isCompatible) onSelectMethod(method.id);
                  }}
                  className={`px-3 py-1.5 rounded text-xs font-semibold transition-all ${
                    isSelected
                      ? 'bg-neuro-cyan text-neuro-bg shadow-sm'
                      : 'neuro-btn-secondary'
                  }`}
                >
                  {isSelected ? 'Active XAI' : 'Select'}
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {/* Segmentation XAI Feature Layer Dropdown */}
      {selectedMethod?.supportsLayerSelection && layers.length > 0 && (
        <div className="neuro-card p-4 space-y-3 bg-neuro-surface-200/50 border border-neuro-border">
          <div className="flex items-center justify-between border-b border-neuro-border pb-2.5">
            <div className="flex items-center gap-2">
              <Sliders className="w-4 h-4 text-neuro-cyan" />
              <h4 className="text-xs font-bold text-neuro-text-primary uppercase tracking-wider">
                Segmentation XAI Target Feature Layer ({modelName})
              </h4>
            </div>
            <span className="text-xs font-mono text-neuro-cyan font-semibold">
              {selectedLayer?.name || 'Default Layer'}
            </span>
          </div>

          <div className="space-y-2 text-xs">
            <label className="text-neuro-text-secondary block">
              Select intermediate tensor stage for segmentation gradient backpropagation:
            </label>
            <select
              value={selectedLayerId}
              onChange={e => onSelectLayer(e.target.value)}
              className="neuro-input w-full font-mono py-2 text-xs bg-neuro-surface"
            >
              {layers.map(layer => (
                <option key={layer.id} value={layer.id}>
                  {layer.name} {layer.isVerified ? '(Verified Latent Stage - Recommended)' : '(Experimental Stage)'}
                </option>
              ))}
            </select>

            {selectedLayer && (
              <div className="p-3 rounded-lg bg-neuro-surface border border-neuro-border space-y-1">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-neuro-text-primary font-mono">{selectedLayer.name}</span>
                  {selectedLayer.isVerified && (
                    <span className="neuro-badge-success text-[10px] flex items-center gap-1">
                      <CheckCircle2 className="w-3 h-3" />
                      Verified
                    </span>
                  )}
                </div>
                <p className="text-[11px] text-neuro-text-secondary">
                  {selectedLayer.description}
                </p>
                {selectedLayer.depthDescription && (
                  <p className="text-[10px] font-mono text-neuro-text-muted">
                    {selectedLayer.depthDescription}
                  </p>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
