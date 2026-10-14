import type { XaiMethod } from '@/types/modelRegistry';
import { Sparkles, Check, Lock } from 'lucide-react';

interface XaiMethodCardProps {
  method: XaiMethod;
  isSelected: boolean;
  isCompatible: boolean;
  onSelect: (methodId: string) => void;
  activeModelName: string;
}

export default function XaiMethodCard({
  method,
  isSelected,
  isCompatible,
  onSelect,
  activeModelName,
}: XaiMethodCardProps) {
  // SVG illustrative preview graphics
  const renderIllustrativePreview = () => {
    if (method.id === 'gradcam') {
      return (
        <div className="relative w-full h-16 rounded bg-black/60 overflow-hidden flex items-center justify-center border border-neuro-border/60">
          <div className="absolute inset-0 bg-radial from-red-500/80 via-yellow-400/50 via-cyan-400/20 to-transparent blur-md transform translate-x-2 -translate-y-1" />
          <span className="relative z-10 text-[9px] font-mono text-neuro-text-muted bg-black/70 px-1.5 py-0.5 rounded">
            Illustrative Saliency
          </span>
        </div>
      );
    }
    if (method.id === 'gradcam_pp') {
      return (
        <div className="relative w-full h-16 rounded bg-black/60 overflow-hidden flex items-center justify-center border border-neuro-border/60">
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="w-8 h-8 rounded-full bg-red-500/80 blur-sm transform -translate-x-3" />
            <div className="w-6 h-6 rounded-full bg-yellow-400/80 blur-sm transform translate-x-3 translate-y-1" />
          </div>
          <span className="relative z-10 text-[9px] font-mono text-neuro-text-muted bg-black/70 px-1.5 py-0.5 rounded">
            Illustrative Multi-Focal
          </span>
        </div>
      );
    }
    return (
      <div className="relative w-full h-16 rounded bg-black/60 overflow-hidden flex items-center justify-center border border-neuro-border/60">
        <div className="absolute inset-0 opacity-40 bg-[radial-gradient(#2dd4bf_1px,transparent_1px)] [background-size:6px_6px]" />
        <span className="relative z-10 text-[9px] font-mono text-neuro-text-muted bg-black/70 px-1.5 py-0.5 rounded">
          Illustrative Path Integral
        </span>
      </div>
    );
  };

  return (
    <div
      onClick={() => isCompatible && onSelect(method.id)}
      className={`neuro-card p-4 transition-all duration-200 border flex flex-col justify-between relative ${
        isCompatible ? 'cursor-pointer' : 'cursor-not-allowed opacity-60'
      } ${
        isSelected
          ? 'border-neuro-accent bg-neuro-accent-muted/20 shadow-neuro-glow ring-1 ring-neuro-accent'
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
                  ? 'bg-neuro-accent text-neuro-bg'
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

          <div className="flex items-center gap-1">
            {isCompatible ? (
              <span className="neuro-badge-success text-[10px]">Compatible</span>
            ) : (
              <span className="neuro-badge-warning text-[10px] flex items-center gap-1">
                <Lock className="w-3 h-3" />
                Incompatible
              </span>
            )}
            {isSelected && (
              <div className="w-5 h-5 rounded-full bg-neuro-accent text-neuro-bg flex items-center justify-center ml-1">
                <Check className="w-3.5 h-3.5 stroke-[3]" />
              </div>
            )}
          </div>
        </div>

        {/* Illustrative Thumbnail */}
        <div className="my-2.5">
          {renderIllustrativePreview()}
        </div>

        <p className="text-xs text-neuro-text-secondary leading-relaxed mt-2">
          {method.description}
        </p>

        {!isCompatible && (
          <div className="mt-2.5 p-2 rounded bg-amber-500/10 border border-amber-500/20 text-[11px] text-amber-300">
            {method.name} is not currently supported by {activeModelName}.
          </div>
        )}
      </div>

      <div className="pt-3 mt-4 border-t border-neuro-border/60 flex items-center justify-between text-xs">
        <span className="text-[11px] text-neuro-text-muted">
          {method.supportsLayerSelection ? 'Supports layer tuning' : 'Direct input attribution'}
        </span>

        <button
          type="button"
          disabled={!isCompatible}
          onClick={e => {
            e.stopPropagation();
            if (isCompatible) onSelect(method.id);
          }}
          className={`px-3 py-1.5 rounded text-xs font-semibold transition-all ${
            isSelected
              ? 'bg-neuro-accent text-neuro-bg shadow-sm'
              : isCompatible
              ? 'neuro-btn-secondary'
              : 'opacity-40 cursor-not-allowed bg-neuro-surface-200 text-neuro-text-disabled'
          }`}
        >
          {isSelected ? 'Selected' : isCompatible ? 'Select Method' : 'Incompatible'}
        </button>
      </div>
    </div>
  );
}
