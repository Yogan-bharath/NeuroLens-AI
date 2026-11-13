import type { PerClassMetrics } from '@/types/model';
import { formatPercentage } from '@/lib/utils';

interface PerformanceChartsProps {
  metrics: PerClassMetrics[];
}

export default function PerformanceCharts({ metrics }: PerformanceChartsProps) {
  return (
    <div className="space-y-4">
      <div className="space-y-4">
        {metrics.map((item, idx) => (
          <div key={idx} className="p-3 rounded-lg bg-neuro-surface-200/60 border border-neuro-border space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-semibold text-neuro-text-primary">{item.className}</span>
              <span className="text-[11px] text-neuro-text-muted font-mono">
                Support: {item.support} scans
              </span>
            </div>

            <div className="grid grid-cols-3 gap-3 text-xs">
              {/* Precision */}
              <div className="space-y-1">
                <div className="flex justify-between text-[11px]">
                  <span className="text-neuro-text-muted">Precision</span>
                  <span className="font-mono text-neuro-text-primary font-medium">
                    {formatPercentage(item.precision)}
                  </span>
                </div>
                <div className="h-1.5 w-full bg-neuro-surface-300 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-neuro-accent rounded-full"
                    style={{ width: `${item.precision * 100}%` }}
                  />
                </div>
              </div>

              {/* Recall */}
              <div className="space-y-1">
                <div className="flex justify-between text-[11px]">
                  <span className="text-neuro-text-muted">Recall</span>
                  <span className="font-mono text-neuro-text-primary font-medium">
                    {formatPercentage(item.recall)}
                  </span>
                </div>
                <div className="h-1.5 w-full bg-neuro-surface-300 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-neuro-cyan rounded-full"
                    style={{ width: `${item.recall * 100}%` }}
                  />
                </div>
              </div>

              {/* F1 */}
              <div className="space-y-1">
                <div className="flex justify-between text-[11px]">
                  <span className="text-neuro-text-muted">F1-Score</span>
                  <span className="font-mono text-neuro-text-primary font-medium">
                    {formatPercentage(item.f1Score)}
                  </span>
                </div>
                <div className="h-1.5 w-full bg-neuro-surface-300 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-neuro-blue rounded-full"
                    style={{ width: `${item.f1Score * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
