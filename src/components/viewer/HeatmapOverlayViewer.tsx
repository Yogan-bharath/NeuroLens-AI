import { Info } from 'lucide-react';

interface HeatmapOverlayViewerProps {
  methodName: string;
  targetClass: string;
  layerName?: string;
  description?: string;
}

export default function HeatmapOverlayViewer({
  methodName,
  targetClass,
  layerName = 'layer4 (ResNet-50)',
  description,
}: HeatmapOverlayViewerProps) {
  return (
    <div className="space-y-3 p-3 bg-neuro-surface-200/70 border border-neuro-border rounded-lg text-xs">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-1.5 font-medium text-neuro-text-primary">
          <span className="w-2 h-2 rounded-full bg-neuro-accent animate-pulse" />
          <span>Active Method: {methodName}</span>
        </div>
        <span className="neuro-badge-accent">Target: {targetClass}</span>
      </div>

      {/* Heatmap Colormap scale */}
      <div>
        <div className="flex justify-between text-[11px] text-neuro-text-muted mb-1">
          <span>Low Influence (0.0)</span>
          <span>Medium</span>
          <span>High Influence (1.0)</span>
        </div>
        <div className="h-2.5 rounded-full w-full bg-gradient-to-r from-blue-600 via-cyan-400 via-green-400 via-yellow-400 to-red-600 shadow-inner" />
      </div>

      <div className="flex items-center justify-between text-[11px] text-neuro-text-secondary pt-1 border-t border-neuro-border/60">
        <span className="text-neuro-text-muted">Extracted from:</span>
        <span className="font-mono">{layerName}</span>
      </div>

      {description && (
        <div className="flex items-start gap-1.5 p-2 rounded bg-neuro-surface-300/40 text-[11px] text-neuro-text-secondary">
          <Info className="w-3.5 h-3.5 text-neuro-accent flex-shrink-0 mt-0.5" />
          <p className="leading-relaxed">{description}</p>
        </div>
      )}
    </div>
  );
}
