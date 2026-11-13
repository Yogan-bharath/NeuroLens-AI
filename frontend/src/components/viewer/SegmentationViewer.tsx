import type { SegmentationClass } from '@/types/analysis';
import { Layers } from 'lucide-react';

interface SegmentationViewerProps {
  classes: SegmentationClass[];
  description?: string;
}

export default function SegmentationViewer({ classes, description }: SegmentationViewerProps) {
  return (
    <div className="space-y-3 p-3 bg-neuro-surface-200/70 border border-neuro-border rounded-lg text-xs">
      <div className="flex items-center gap-1.5 font-medium text-neuro-text-primary">
        <Layers className="w-3.5 h-3.5 text-neuro-accent" />
        <span>Predicted Segmentation Mask</span>
      </div>

      {/* Class Legend */}
      <div className="space-y-2">
        {classes.map((cls, idx) => (
          <div
            key={idx}
            className="flex items-center justify-between p-2 rounded bg-neuro-surface-300/40 border border-neuro-border/40"
          >
            <div className="flex items-center gap-2">
              <span
                className="w-3.5 h-3.5 rounded-sm flex-shrink-0"
                style={{ backgroundColor: cls.color }}
              />
              <span className="font-medium text-neuro-text-primary">{cls.label}</span>
            </div>
            <div className="flex items-center gap-2 font-mono text-[11px] text-neuro-text-secondary">
              <span>{cls.percentage.toFixed(1)}%</span>
              <span className="text-neuro-text-muted">({cls.pixelCount.toLocaleString()} px)</span>
            </div>
          </div>
        ))}
      </div>

      {description && (
        <p className="text-[11px] text-neuro-text-muted leading-relaxed pt-1 border-t border-neuro-border/60">
          {description}
        </p>
      )}
    </div>
  );
}
