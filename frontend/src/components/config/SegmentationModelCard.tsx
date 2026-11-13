import type { SegmentationModel } from '@/types/modelRegistry';
import { formatPercentage } from '@/lib/utils';
import { Layers, Check, Lock } from 'lucide-react';

interface SegmentationModelCardProps {
  model: SegmentationModel;
  isSelected: boolean;
  onSelect: (modelId: string) => void;
}

export default function SegmentationModelCard({
  model,
  isSelected,
  onSelect,
}: SegmentationModelCardProps) {
  const isAvailable = model.status === 'available';

  return (
    <div
      onClick={() => isAvailable && onSelect(model.id)}
      className={`neuro-card p-4 transition-all duration-200 border flex flex-col justify-between relative ${
        isAvailable ? 'cursor-pointer' : 'cursor-not-allowed opacity-75'
      } ${
        isSelected
          ? 'border-neuro-accent bg-neuro-accent-muted/20 shadow-neuro-glow ring-1 ring-neuro-accent'
          : isAvailable
          ? 'border-neuro-border bg-neuro-surface hover:border-neuro-border-hover hover:bg-neuro-surface-200/40'
          : 'border-neuro-border/60 bg-neuro-surface/60'
      }`}
    >
      <div>
        {/* Card Header */}
        <div className="flex items-start justify-between gap-2 mb-2">
          <div className="flex items-center gap-2.5">
            <div
              className={`w-9 h-9 rounded-lg flex items-center justify-center ${
                isSelected
                  ? 'bg-neuro-accent text-neuro-bg'
                  : 'bg-neuro-surface-200 text-neuro-text-secondary'
              }`}
            >
              <Layers className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h4 className="text-sm font-bold text-neuro-text-primary">{model.name}</h4>
                {model.version && (
                  <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-neuro-surface-200 text-neuro-text-muted">
                    {model.version}
                  </span>
                )}
              </div>
              <p className="text-[11px] text-neuro-text-muted">{model.architecture}</p>
            </div>
          </div>

          <div className="flex items-center gap-1.5">
            {isAvailable ? (
              <span className="neuro-badge-success text-[10px]">Ready</span>
            ) : (
              <span className="neuro-badge-warning text-[10px] flex items-center gap-1">
                <Lock className="w-3 h-3" />
                Coming Soon
              </span>
            )}
            {isSelected && (
              <div className="w-5 h-5 rounded-full bg-neuro-accent text-neuro-bg flex items-center justify-center ml-1">
                <Check className="w-3.5 h-3.5 stroke-[3]" />
              </div>
            )}
          </div>
        </div>

        {/* Description */}
        <p className="text-xs text-neuro-text-secondary leading-relaxed mt-2">
          {model.description}
        </p>

        {/* Disabled explanation if coming soon */}
        {!isAvailable && model.statusReason && (
          <div className="mt-2.5 p-2 rounded bg-amber-500/10 border border-amber-500/20 text-[11px] text-amber-300/90 leading-tight">
            {model.statusReason}
          </div>
        )}

        {/* Classes */}
        <div className="mt-3">
          <span className="text-[10px] uppercase tracking-wider text-neuro-text-muted font-semibold block mb-1">
            Segmented Sub-Regions
          </span>
          <div className="flex flex-wrap gap-1">
            {model.classes.map((cls, idx) => (
              <span
                key={idx}
                className="text-[10px] px-1.5 py-0.5 rounded bg-neuro-surface-200 text-neuro-text-secondary"
              >
                {cls}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Metrics Footer (Dice & IoU) */}
      <div className="pt-3 mt-4 border-t border-neuro-border/60">
        <div className="grid grid-cols-2 gap-2 text-xs mb-3">
          <div className="p-2 rounded bg-neuro-surface-200/70 border border-neuro-border/40">
            <span className="text-[10px] text-neuro-text-muted uppercase block font-semibold">Dice Coefficient</span>
            <span className="font-mono text-sm font-bold text-neuro-text-primary mt-0.5 block">
              {model.metrics?.diceCoefficient ? (
                formatPercentage(model.metrics.diceCoefficient)
              ) : (
                <span className="text-neuro-text-disabled text-xs italic font-normal">Not evaluated</span>
              )}
            </span>
          </div>

          <div className="p-2 rounded bg-neuro-surface-200/70 border border-neuro-border/40">
            <span className="text-[10px] text-neuro-text-muted uppercase block font-semibold">IoU (Jaccard)</span>
            <span className="font-mono text-sm font-bold text-neuro-text-primary mt-0.5 block">
              {model.metrics?.iouScore ? (
                formatPercentage(model.metrics.iouScore)
              ) : (
                <span className="text-neuro-text-disabled text-xs italic font-normal">Not evaluated</span>
              )}
            </span>
          </div>
        </div>

        <button
          type="button"
          disabled={!isAvailable}
          onClick={e => {
            e.stopPropagation();
            if (isAvailable) onSelect(model.id);
          }}
          className={`w-full py-1.5 rounded text-xs font-semibold transition-all ${
            isSelected
              ? 'bg-neuro-accent text-neuro-bg shadow-sm'
              : isAvailable
              ? 'neuro-btn-secondary'
              : 'opacity-40 cursor-not-allowed bg-neuro-surface-200 text-neuro-text-disabled'
          }`}
        >
          {isSelected ? 'Selected Model' : isAvailable ? 'Select Model' : 'Integration Pending'}
        </button>
      </div>
    </div>
  );
}
