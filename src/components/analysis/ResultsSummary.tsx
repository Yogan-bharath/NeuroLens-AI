import type { AnalysisResult } from '@/types/analysis';
import ExportReportButton from '../common/ExportReportButton';
import { Brain, Sparkles, CheckCircle2 } from 'lucide-react';
import { formatPercentage } from '@/lib/utils';

interface ResultsSummaryProps {
  analysis: AnalysisResult;
  selectedMethod: string;
}

export default function ResultsSummary({ analysis, selectedMethod }: ResultsSummaryProps) {
  return (
    <div className="neuro-card flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-neuro-accent-muted flex items-center justify-center">
            <Brain className="w-4 h-4 text-neuro-accent-bright" />
          </div>
          <div>
            <p className="text-[10px] uppercase font-semibold text-neuro-text-muted">Prediction</p>
            <p className="text-sm font-bold text-neuro-text-primary">
              {analysis.predictedClass} ({analysis.confidence ? formatPercentage(analysis.confidence) : '—'})
            </p>
          </div>
        </div>

        <div className="h-6 w-[1px] bg-neuro-border hidden sm:block" />

        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-neuro-surface-200 flex items-center justify-center">
            <Sparkles className="w-4 h-4 text-neuro-cyan" />
          </div>
          <div>
            <p className="text-[10px] uppercase font-semibold text-neuro-text-muted">Active Method</p>
            <p className="text-xs font-semibold text-neuro-text-primary uppercase">
              {selectedMethod}
            </p>
          </div>
        </div>

        <div className="h-6 w-[1px] bg-neuro-border hidden sm:block" />

        <div className="flex items-center gap-1.5 text-xs text-neuro-text-secondary">
          <CheckCircle2 className="w-4 h-4 text-neuro-success" />
          <span>Segmentation Available</span>
        </div>
      </div>

      <ExportReportButton analysis={analysis} />
    </div>
  );
}
