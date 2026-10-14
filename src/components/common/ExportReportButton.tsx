import { useState } from 'react';
import type { AnalysisResult } from '@/types/analysis';
import { FileDown, Printer, Check, X } from 'lucide-react';
import { formatDate, formatPercentage } from '@/lib/utils';

interface ExportReportButtonProps {
  analysis: AnalysisResult;
}

export default function ExportReportButton({ analysis }: ExportReportButtonProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  const handlePrint = () => {
    window.print();
  };

  const handleCopyMarkdown = () => {
    const md = `# NeuroLens AI — Analysis Report
**Analysis ID:** ${analysis.id}
**File Name:** ${analysis.fileName}
**Date:** ${formatDate(analysis.createdAt)}
**Model:** ${analysis.metadata?.modelName || 'NeuroClassifier'} (${analysis.metadata?.modelVersion || 'v2.1.0'})
**Predicted Class:** ${analysis.predictedClass}
**Confidence:** ${analysis.confidence ? formatPercentage(analysis.confidence) : 'N/A'}

## Class Probability Distribution
${(analysis.classification || [])
  .map(c => `- **${c.label}:** ${formatPercentage(c.score)}${c.isTopPrediction ? ' *(Top Prediction)*' : ''}`)
  .join('\n')}

## Explainability (XAI)
- **Grad-CAM Layer:** ${analysis.explanations[0]?.layerName || 'layer4'}
- **Segmentation Available:** ${analysis.segmentation ? 'Yes' : 'No'}

> **DISCLAIMER:** Research and educational demonstration only. Not for clinical diagnosis.
`;
    navigator.clipboard.writeText(md);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        className="neuro-btn-secondary text-xs py-1.5 px-3"
        title="Export Analysis Report"
      >
        <FileDown className="w-3.5 h-3.5" />
        <span>Export Report</span>
      </button>

      {/* Modal Dialog */}
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm animate-fade-in">
          <div className="relative w-full max-w-2xl bg-neuro-surface border border-neuro-border rounded-xl shadow-2xl overflow-hidden max-h-[90vh] flex flex-col">
            {/* Modal Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-neuro-border bg-neuro-surface-200">
              <div className="flex items-center gap-2">
                <FileDown className="w-5 h-5 text-neuro-accent" />
                <h3 className="text-base font-semibold text-neuro-text-primary">
                  Analysis Research Report
                </h3>
              </div>
              <button
                onClick={() => setIsOpen(false)}
                className="p-1 rounded-md text-neuro-text-muted hover:text-neuro-text-primary hover:bg-neuro-surface-300 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Body / Report Preview */}
            <div className="p-6 overflow-y-auto space-y-4 font-sans text-xs scrollbar-thin">
              <div className="p-4 rounded-lg bg-neuro-surface-200 border border-neuro-border space-y-2">
                <div className="flex justify-between items-center border-b border-neuro-border pb-2">
                  <span className="text-sm font-bold text-neuro-text-primary">NeuroLens AI Analysis Summary</span>
                  <span className="text-[10px] uppercase tracking-wider px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 font-semibold">
                    Research Demonstration
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-neuro-text-secondary pt-1">
                  <div><strong className="text-neuro-text-primary">ID:</strong> {analysis.id}</div>
                  <div><strong className="text-neuro-text-primary">Date:</strong> {formatDate(analysis.createdAt)}</div>
                  <div><strong className="text-neuro-text-primary">File:</strong> {analysis.fileName}</div>
                  <div><strong className="text-neuro-text-primary">Model:</strong> {analysis.metadata?.modelName} {analysis.metadata?.modelVersion}</div>
                </div>
              </div>

              {/* Classification Summary */}
              <div className="p-4 rounded-lg bg-neuro-surface-200 border border-neuro-border space-y-2">
                <h4 className="font-semibold text-neuro-text-primary uppercase tracking-wider text-[11px]">
                  Classification Outcome
                </h4>
                <div className="flex items-center justify-between p-2 rounded bg-neuro-surface-300">
                  <span className="font-medium text-neuro-text-primary">Primary Prediction</span>
                  <span className="text-sm font-bold text-neuro-accent-bright">
                    {analysis.predictedClass} ({analysis.confidence ? formatPercentage(analysis.confidence) : '—'})
                  </span>
                </div>
                <div className="space-y-1 pt-1">
                  {(analysis.classification || []).map((c, i) => (
                    <div key={i} className="flex justify-between text-neuro-text-secondary">
                      <span>{c.label}</span>
                      <span className="font-mono">{formatPercentage(c.score)}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Explainability Summary */}
              <div className="p-4 rounded-lg bg-neuro-surface-200 border border-neuro-border space-y-1.5">
                <h4 className="font-semibold text-neuro-text-primary uppercase tracking-wider text-[11px]">
                  Explainability Verification
                </h4>
                <p className="text-neuro-text-secondary">
                  Grad-CAM and Grad-CAM++ convolutional saliency heatmaps successfully synthesized for target class <strong>{analysis.predictedClass}</strong>.
                </p>
                {analysis.segmentation && (
                  <p className="text-neuro-text-secondary">
                    Segmentation mask generated across <strong>{analysis.segmentation.classes.length}</strong> classes.
                  </p>
                )}
              </div>

              {/* Warning Disclaimer */}
              <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 text-[11px] text-amber-300 leading-relaxed">
                <strong>Notice:</strong> This synthetic report is intended for research and educational presentation purposes only. It is not an official medical record or certified radiological diagnosis.
              </div>
            </div>

            {/* Modal Footer Actions */}
            <div className="flex items-center justify-between px-6 py-3 border-t border-neuro-border bg-neuro-surface-200">
              <button
                onClick={handleCopyMarkdown}
                className="neuro-btn-secondary text-xs"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-neuro-success" /> : null}
                <span>{copied ? 'Copied Markdown' : 'Copy as Markdown'}</span>
              </button>

              <div className="flex items-center gap-2">
                <button
                  onClick={handlePrint}
                  className="neuro-btn-primary text-xs"
                >
                  <Printer className="w-3.5 h-3.5" />
                  <span>Print Report</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
