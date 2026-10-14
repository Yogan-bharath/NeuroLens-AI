import { useState } from 'react';
import type { ClassificationModel } from '@/types/modelRegistry';
import { formatPercentage } from '@/lib/utils';
import { X, ArrowUpDown, Info } from 'lucide-react';

interface ModelComparisonTableProps {
  models: ClassificationModel[];
  onClose: () => void;
  onSelectModel: (modelId: string) => void;
  currentSelectedId: string;
}

type SortField = 'name' | 'testAccuracy' | 'macroF1' | 'macroPrecision' | 'macroRecall' | 'inferenceTimeMs';

export default function ModelComparisonTable({
  models,
  onClose,
  onSelectModel,
  currentSelectedId,
}: ModelComparisonTableProps) {
  const [sortField, setSortField] = useState<SortField>('testAccuracy');
  const [sortAsc, setSortAsc] = useState(false);

  const handleSort = (field: SortField) => {
    if (sortField === field) {
      setSortAsc(prev => !prev);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  const sortedModels = [...models].sort((a, b) => {
    if (sortField === 'name') {
      return sortAsc ? a.name.localeCompare(b.name) : b.name.localeCompare(a.name);
    }

    const valA = a.metrics?.[sortField] ?? -1;
    const valB = b.metrics?.[sortField] ?? -1;

    return sortAsc ? valA - valB : valB - valA;
  });

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/75 backdrop-blur-sm animate-fade-in">
      <div className="relative w-full max-w-4xl bg-neuro-surface border border-neuro-border rounded-xl shadow-2xl overflow-hidden max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-neuro-border bg-neuro-surface-200">
          <div>
            <h3 className="text-base font-semibold text-neuro-text-primary">
              Classification Models Benchmark Comparison
            </h3>
            <p className="text-xs text-neuro-text-muted mt-0.5">
              Verified metric benchmarks on held-out brain MRI test partitions.
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-md text-neuro-text-muted hover:text-neuro-text-primary hover:bg-neuro-surface-300 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Table */}
        <div className="p-6 overflow-y-auto space-y-4 text-xs scrollbar-thin">
          <div className="overflow-x-auto rounded-lg border border-neuro-border">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-neuro-surface-200/90 text-neuro-text-muted text-[11px] font-semibold border-b border-neuro-border">
                  <th className="py-3 px-4 cursor-pointer hover:text-neuro-text-primary" onClick={() => handleSort('name')}>
                    <div className="flex items-center gap-1">
                      <span>Model Architecture</span>
                      <ArrowUpDown className="w-3 h-3" />
                    </div>
                  </th>
                  <th className="py-3 px-3 cursor-pointer hover:text-neuro-text-primary text-center" onClick={() => handleSort('testAccuracy')}>
                    <div className="flex items-center justify-center gap-1">
                      <span>Test Accuracy</span>
                      <ArrowUpDown className="w-3 h-3" />
                    </div>
                  </th>
                  <th className="py-3 px-3 cursor-pointer hover:text-neuro-text-primary text-center" onClick={() => handleSort('macroPrecision')}>
                    <div className="flex items-center justify-center gap-1">
                      <span>Precision</span>
                      <ArrowUpDown className="w-3 h-3" />
                    </div>
                  </th>
                  <th className="py-3 px-3 cursor-pointer hover:text-neuro-text-primary text-center" onClick={() => handleSort('macroRecall')}>
                    <div className="flex items-center justify-center gap-1">
                      <span>Recall</span>
                      <ArrowUpDown className="w-3 h-3" />
                    </div>
                  </th>
                  <th className="py-3 px-3 cursor-pointer hover:text-neuro-text-primary text-center" onClick={() => handleSort('macroF1')}>
                    <div className="flex items-center justify-center gap-1">
                      <span>Macro F1</span>
                      <ArrowUpDown className="w-3 h-3" />
                    </div>
                  </th>
                  <th className="py-3 px-3 text-center">Inference Time</th>
                  <th className="py-3 px-3 text-center">Status</th>
                  <th className="py-3 px-4 text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-neuro-border">
                {sortedModels.map(model => {
                  const isCurrent = model.id === currentSelectedId;
                  const isAvailable = model.status === 'available';

                  return (
                    <tr
                      key={model.id}
                      className={`hover:bg-neuro-surface-200/50 transition-colors ${
                        isCurrent ? 'bg-neuro-accent-muted/20' : ''
                      }`}
                    >
                      <td className="py-3 px-4">
                        <div className="font-semibold text-neuro-text-primary">{model.name}</div>
                        <div className="text-[10px] text-neuro-text-muted mt-0.5">{model.architecture}</div>
                        <div className="text-[10px] font-mono text-neuro-text-disabled">{model.parametersCount} params</div>
                      </td>

                      <td className="py-3 px-3 text-center font-mono font-medium">
                        {model.metrics?.testAccuracy ? (
                          <span className="text-neuro-accent-bright font-semibold">
                            {formatPercentage(model.metrics.testAccuracy)}
                          </span>
                        ) : (
                          <span className="text-neuro-text-disabled italic">Not evaluated</span>
                        )}
                      </td>

                      <td className="py-3 px-3 text-center font-mono">
                        {model.metrics?.macroPrecision ? (
                          formatPercentage(model.metrics.macroPrecision)
                        ) : (
                          <span className="text-neuro-text-disabled italic">Not evaluated</span>
                        )}
                      </td>

                      <td className="py-3 px-3 text-center font-mono">
                        {model.metrics?.macroRecall ? (
                          formatPercentage(model.metrics.macroRecall)
                        ) : (
                          <span className="text-neuro-text-disabled italic">Not evaluated</span>
                        )}
                      </td>

                      <td className="py-3 px-3 text-center font-mono font-medium">
                        {model.metrics?.macroF1 ? (
                          formatPercentage(model.metrics.macroF1)
                        ) : (
                          <span className="text-neuro-text-disabled italic">Not evaluated</span>
                        )}
                      </td>

                      <td className="py-3 px-3 text-center font-mono text-neuro-text-secondary">
                        {model.metrics?.inferenceTimeMs ? (
                          `${model.metrics.inferenceTimeMs} ms`
                        ) : (
                          <span className="text-neuro-text-disabled italic">—</span>
                        )}
                      </td>

                      <td className="py-3 px-3 text-center">
                        {isAvailable ? (
                          <span className="neuro-badge-success">Available</span>
                        ) : (
                          <span className="neuro-badge-warning">Coming Soon</span>
                        )}
                      </td>

                      <td className="py-3 px-4 text-right">
                        <button
                          type="button"
                          disabled={!isAvailable}
                          onClick={() => {
                            onSelectModel(model.id);
                            onClose();
                          }}
                          className={`px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                            isCurrent
                              ? 'bg-neuro-accent text-neuro-bg font-semibold'
                              : isAvailable
                              ? 'neuro-btn-secondary'
                              : 'opacity-40 cursor-not-allowed text-neuro-text-disabled bg-neuro-surface-200'
                          }`}
                        >
                          {isCurrent ? 'Selected' : isAvailable ? 'Choose' : 'Unavailable'}
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          <div className="flex items-start gap-2 p-3 rounded-lg bg-neuro-surface-200 border border-neuro-border text-[11px] text-neuro-text-muted">
            <Info className="w-4 h-4 text-neuro-accent flex-shrink-0 mt-0.5" />
            <p>
              <strong>Data Transparency Notice:</strong> Metrics displayed are derived from verified cross-validation benchmarks. Unintegrated or in-training models are strictly labeled as &quot;Not evaluated&quot;. NeuroLens AI does not declare architectural superiority without standardized, identical evaluation split measurements.
            </p>
          </div>
        </div>

        <div className="flex justify-end px-6 py-3 border-t border-neuro-border bg-neuro-surface-200">
          <button onClick={onClose} className="neuro-btn-secondary text-xs">
            Close Comparison
          </button>
        </div>
      </div>
    </div>
  );
}
