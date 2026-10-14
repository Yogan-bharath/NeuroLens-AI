import { useState } from 'react';
import type { ClassificationModel } from '@/types/modelRegistry';
import ClassificationModelCard from './ClassificationModelCard';
import ModelComparisonTable from './ModelComparisonTable';
import { Columns3 } from 'lucide-react';

interface ClassificationModelSelectorProps {
  models: ClassificationModel[];
  selectedModelId: string;
  onSelectModel: (modelId: string) => void;
}

export default function ClassificationModelSelector({
  models,
  selectedModelId,
  onSelectModel,
}: ClassificationModelSelectorProps) {
  const [showComparison, setShowComparison] = useState(false);

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-neuro-text-primary uppercase tracking-wider text-xs">
            Step 2: Classification Model
          </h3>
          <p className="text-xs text-neuro-text-muted mt-0.5">
            Select the neural network backbone for multi-class pathological inference.
          </p>
        </div>

        <button
          type="button"
          onClick={() => setShowComparison(true)}
          className="neuro-btn-secondary text-xs py-1.5 px-3 flex items-center gap-1.5"
        >
          <Columns3 className="w-3.5 h-3.5 text-neuro-accent" />
          <span>Compare Models</span>
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        {models.map(model => (
          <ClassificationModelCard
            key={model.id}
            model={model}
            isSelected={selectedModelId === model.id}
            onSelect={onSelectModel}
          />
        ))}
      </div>

      {showComparison && (
        <ModelComparisonTable
          models={models}
          onClose={() => setShowComparison(false)}
          onSelectModel={onSelectModel}
          currentSelectedId={selectedModelId}
        />
      )}
    </div>
  );
}
