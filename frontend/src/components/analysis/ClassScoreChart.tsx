import type { ClassificationScore } from '@/types/analysis';
import { formatPercentage } from '@/lib/utils';

interface ClassScoreChartProps {
  scores: ClassificationScore[];
}

export default function ClassScoreChart({ scores }: ClassScoreChartProps) {
  return (
    <div className="space-y-3">
      {scores.map((item, idx) => {
        const percent = item.score * 100;
        const isTop = item.isTopPrediction;

        return (
          <div key={idx} className="space-y-1">
            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-1.5">
                <span
                  className={`w-2 h-2 rounded-full ${
                    isTop ? 'bg-neuro-accent animate-pulse' : 'bg-neuro-surface-400'
                  }`}
                />
                <span className={`font-medium ${isTop ? 'text-neuro-text-primary' : 'text-neuro-text-secondary'}`}>
                  {item.label}
                </span>
                {isTop && (
                  <span className="text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.2 rounded bg-neuro-accent/20 text-neuro-accent-bright">
                    Top Prediction
                  </span>
                )}
              </div>
              <span className="font-mono text-xs font-semibold text-neuro-text-primary">
                {formatPercentage(item.score)}
              </span>
            </div>

            {/* Horizontal Bar */}
            <div className="h-2 w-full bg-neuro-surface-300 rounded-full overflow-hidden p-0.5">
              <div
                className={`h-full rounded-full transition-all duration-700 ease-out ${
                  isTop
                    ? 'bg-gradient-to-r from-neuro-accent to-neuro-cyan'
                    : 'bg-neuro-surface-400'
                }`}
                style={{ width: `${Math.max(percent, 2)}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
